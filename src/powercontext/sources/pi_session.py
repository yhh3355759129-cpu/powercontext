# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline Pi v3 branch evidence with lifecycle-owned durable progress."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from powercontext.errors import SourceConflictError
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import SourceDefinition, SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

_RECORD = TypeAdapter(dict[str, JsonValue])


class PiSessionInput(BaseModel):
    """Complete UTF-8 JSONL archive and optional selected branch endpoint."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event together with its complete original JSON record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_compact(value).encode("utf-8")).hexdigest()


def _event_identity(session_id: str, event_id: str) -> str:
    # A structured pair avoids delimiter ambiguity, and hashing bounds long IDs.
    return f"pi:{_digest([session_id, event_id])}"


def _required_text(record: dict[str, JsonValue], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pi {key} must be a non-blank string")  # noqa: TRY003
    return value


def _text(value: JsonValue, *, tool_calls: bool = False) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if not isinstance(value, list):
        raise ValueError("Pi content must be a string or a block list")  # noqa: TRY003, TRY004
    parts = []
    for block in value:
        if not isinstance(block, dict):
            raise ValueError("Pi content blocks must be objects")  # noqa: TRY003, TRY004
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("Pi text blocks must contain string text")  # noqa: TRY003
            parts.append(text)
        elif tool_calls and block.get("type") == "toolCall":
            name, call_id = _required_text(block, "name"), _required_text(block, "id")
            parts.append(f"[tool_call {name} {call_id}] {_compact(block.get('arguments', {}))}")
    return "\n".join(parts)


def _event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = _required_text(record, "type")
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message must be an object")  # noqa: TRY003
        role = _required_text(message, "role")
        if role == "bashExecution":
            command, output = message.get("command", ""), message.get("output", "")
            if not isinstance(command, str) or not isinstance(output, str):
                raise ValueError("Pi bash command and output must be strings")  # noqa: TRY003
            body = f"$ {command}\n{output}" if command or output else ""
        else:
            body = _text(message.get("content"), tool_calls=True)
            if role == "toolResult" and body:
                body = f"[tool_result {_required_text(message, 'toolName')} {_required_text(message, 'toolCallId')}] {body}"
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        body = record.get("summary", "")
        if not isinstance(body, str):
            raise ValueError("Pi summaries must be strings")  # noqa: TRY003
    elif event_type == "custom_message":
        role, body = "custom", _text(record.get("content"))
    else:
        return None
    if not body:
        return None
    parent = record["parentId"]
    return PiEventInput(
        session_id=session_id,
        event_id=_required_text(record, "id"),
        parent_id=parent if isinstance(parent, str) else None,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


@dataclass(frozen=True)
class _Session:
    session_id: str
    records: dict[str, dict[str, JsonValue]]
    events: tuple[PiEventInput, ...]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value} is not supported")  # noqa: TRY003


def _parse(value: PiSessionInput) -> _Session:
    records = [
        _RECORD.validate_python(json.loads(line, parse_constant=_reject_constant))
        for line in value.content.decode("utf-8-sig").split("\n")
        if line.strip()
    ]
    if (
        not records
        or records[0].get("type") != "session"
        or type(records[0].get("version")) is not int
        or records[0]["version"] != 3
    ):
        raise ValueError("Pi archives must begin with a version 3 session header")  # noqa: TRY003
    session_id = _required_text(records[0], "id")
    by_id: dict[str, dict[str, JsonValue]] = {}
    visible: dict[str, PiEventInput] = {}
    for record in records[1:]:
        event_id = _required_text(record, "id")
        if event_id in by_id:
            raise ValueError("Pi event IDs must be unique")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError("Pi nodes must declare parentId")  # noqa: TRY003
        parent = record["parentId"]
        # Parent-before-child validation also rejects all cycles and self links.
        if parent is not None and (not isinstance(parent, str) or parent not in by_id):
            raise ValueError("Pi parent must refer to an earlier node")  # noqa: TRY003
        event = _event(session_id, record)
        if event is not None:
            visible[event_id] = event
        by_id[event_id] = record
    leaf = value.leaf_id if value.leaf_id is not None else next(reversed(by_id), None)
    if leaf is not None and leaf not in by_id:
        raise ValueError("Pi selected leaf does not exist")  # noqa: TRY003
    chain = []
    while leaf is not None:
        if leaf in visible:
            chain.append(visible[leaf])
        parent = by_id[leaf]["parentId"]
        leaf = parent if isinstance(parent, str) else None
    return _Session(session_id, by_id, tuple(reversed(chain)))


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Validate every record before returning the selected root-to-leaf evidence."""
    return _parse(value).events


class PiEventSource(Source):
    """Captured Pi evidence; large tool results are never truncated."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class _PiTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class _PiEventDefinition:
    name = "pi-event"
    version = "1"
    input_class = PiEventInput
    source_class = PiEventSource
    projections: tuple[SourceProjection[PiEventSource], ...] = (_PiTextEvidenceProjection(),)

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=_event_identity(value.session_id, value.event_id),
            materialization=SourceMaterialization.CAPTURED,
            **value.model_dump(),
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput.model_validate(source.model_dump())


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = _PiEventDefinition()


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    imported: dict[str, str] = Field(default_factory=dict)


class PiSessionConnector:
    """Import unseen branch events; the lifecycle alone commits durable progress."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        parsed = _parse(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        previous = (
            _Checkpoint.model_validate(session.checkpoint)
            if session.checkpoint is not None
            else _Checkpoint(session_id=parsed.session_id)
        )
        if previous.session_id != parsed.session_id:
            raise ValueError("Pi session identity changed for this binding")  # noqa: TRY003
        # Check *all* imported records, including events on an unselected branch.
        for event_id, fingerprint in previous.imported.items():
            record = parsed.records.get(event_id)
            if record is None or _digest(record) != fingerprint:
                raise ValueError("previously imported Pi event was removed or rewritten")  # noqa: TRY003
        imported = dict(previous.imported)
        for event in parsed.events:
            if event.event_id in imported:
                continue
            try:
                outcome = await session.submit(_event_identity(event.session_id, event.event_id), "pi-event", event)
            except SourceConflictError as error:
                # A prior interrupted run may have stored this event before its
                # checkpoint committed. Runtime immutability still guards it.
                raise ValueError("previously captured Pi event conflicts with the input") from error  # noqa: TRY003
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                return ConnectorRunCompletion(status=ConnectorRunStatus.INCOMPLETE)
            imported[event.event_id] = _digest(event.record)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=_Checkpoint(session_id=parsed.session_id, imported=imported).model_dump(mode="json"),
        )
