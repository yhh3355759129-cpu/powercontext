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

"""Pi parsing and real Runtime recovery without executing captured commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CommitConnectorCheckpoint,
    GetMemoryEntryRequest,
    SearchMemoryRequest,
    SubmitSourceObservation,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.errors import ConnectorSubmissionRejectedError
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    ConnectorBinding,
    ConnectorRunStatus,
    SourceDefinitionRegistry,
    SourceMaterialization,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
)


def archive(records, *, session_id="session", bom=False):
    rows = [{"type": "session", "version": 3, "id": session_id}, *records]
    raw = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    return b"\xef\xbb\xbf" + raw if bom else raw


def message(event_id, parent=None, text="evidence", role="user"):
    return {"type": "message", "id": event_id, "parentId": parent, "message": {"role": role, "content": text}}


def parse(records, leaf=None, **kwargs):
    return parse_pi_session(PiSessionInput(content=archive(records, **kwargs), leaf_id=leaf))


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\xef\xbb\xbf",
        b"\xff",
        b"{}",
        b"null",
        b"[]",
        b"1",
        b'"text"',
        b'{"type":"session","version":2,"id":"s"}',
        b'{"type":"session","version":3.0,"id":"s"}',
        b'{"type":"session","version":3,"id":""}',
        b'{"type":"session","version":3,"id":4}',
        archive([message("ok")]) + b"{bad json}\n",
        archive([message("ok")]) + b"\xc3\n",
        archive([message("ok")]) + b"null\n",
        archive([message("ok")]) + b'{"id":"bad","parentId":null,"type":"custom_message","content":NaN}\n',
    ],
)
def test_invalid_archives_never_return_a_valid_prefix(raw):
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=raw, leaf_id="ok"))


@pytest.mark.parametrize(
    "bad",
    [
        message("ok"),
        message("self", "self"),
        message("future", "later"),
        message("missing", "absent"),
        {"type": "model_change", "id": "missing-parent"},
        {"type": "model_change", "id": "", "parentId": None},
        {"type": "model_change", "id": "bad-parent", "parentId": []},
        {"type": "message", "id": "bad-message", "parentId": None, "message": []},
        message("bad-text", text=[{"type": "text", "text": 123}]),
    ],
)
def test_damaged_unselected_records_are_validated(bad):
    with pytest.raises(ValueError):
        parse([message("ok"), bad], "ok")


def test_cycles_and_missing_leaf_fail_but_header_only_is_valid():
    with pytest.raises(ValueError):
        parse([message("a", "b"), message("b", "a")])
    with pytest.raises(ValueError):
        parse([message("a")], "absent")
    assert parse([]) == ()
    with pytest.raises(ValueError):
        parse([], "absent")


def test_all_visible_formats_and_control_bridges_preserve_original_text():
    tool = message(
        "tool",
        "root",
        [
            {"type": "text", "text": " leading \n"},
            {"type": "thinking", "thinking": "private"},
            {"type": "toolCall", "name": "read", "id": "call", "arguments": {"z": "文字", "a": [1, False]}},
            {"type": "image", "data": "not text"},
            {"type": "text", "text": " trailing "},
        ],
        "assistant",
    )
    result = message("result", "tool", [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}], "toolResult")
    result["message"].update(toolName="read", toolCallId="call")
    bash = message("bash", "bridge", [], "bashExecution")
    bash["message"].update(command="echo never-execute", output=" exact output \n")
    records = [
        message("root", text="before compaction\n"),
        tool,
        result,
        {"type": "model_change", "id": "bridge", "parentId": "result"},
        bash,
        {"type": "compaction", "id": "compact", "parentId": "bash", "summary": " retained summary "},
        {"type": "branch_summary", "id": "branch", "parentId": "compact", "summary": "branch summary"},
        {"type": "custom_message", "id": "custom", "parentId": "branch", "content": "\u2028Unicode\u0085 text "},
        message(
            "invisible",
            "custom",
            [{"type": "thinking", "thinking": "private"}, {"type": "image", "data": "image"}],
            "assistant",
        ),
    ]
    events = parse(records, bom=True, session_id="会话")
    assert [event.content for event in events] == [
        "user: before compaction\n",
        'assistant:  leading \n\n[tool_call read call] {"a":[1,false],"z":"文字"}\n trailing ',
        "toolResult: [tool_result read call] one\ntwo",
        "bashExecution: $ echo never-execute\n exact output \n",
        "compaction:  retained summary ",
        "branch_summary: branch summary",
        "custom: \u2028Unicode\u0085 text ",
    ]
    assert [event.event_id for event in events] == ["root", "tool", "result", "bash", "compact", "branch", "custom"]
    assert events[3].parent_id == "bridge"
    by_id = {record["id"]: record for record in records}
    assert all(event.record == by_id[event.event_id] for event in events)
    assert parse(records, "tool", session_id="会话") == events[:2]
    assert parse([message("empty", text="")]) == ()
    assert parse([message("spaces", text=" \t ")])[0].content == "user:  \t "


def test_identity_is_independent_of_text_leaf_order_and_delimiters():
    async def scenario():
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        sources = []
        for session_id, event_id in [("a:b", "c"), ("a", "b:c"), ("%", ":"), ("文字" * 200, "事件" * 200)]:
            event = parse([message(event_id)], session_id=session_id)[0]
            source = await registry.resolve(event)
            assert source.materialization is SourceMaterialization.CAPTURED
            assert len(source.name) <= 256
            assert await registry.read(source) == event
            edited = parse([message(event_id, text="edited")], session_id=session_id)[0]
            assert (await registry.resolve(edited)).name == source.name
            evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
            assert evidence.source_id == source.name and evidence.content == event.content
            assert evidence.metadata == {
                key: event.model_dump()[key] for key in ("session_id", "event_id", "parent_id", "event_type", "role")
            }
            sources.append(source.name)
        assert len(set(sources)) == len(sources)

    asyncio.run(scenario())


class RuntimeSink:
    def __init__(self, runtime, *, fail_event=None, mode="raise"):
        self.runtime, self.fail_event, self.mode = runtime, fail_event, mode
        self.registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))

    async def submit(self, binding, item_id, definition_name, value):
        if value.event_id == self.fail_event and self.mode != "after":
            if self.mode == "reject":
                raise ConnectorSubmissionRejectedError("injected rejection")  # noqa: TRY003
            raise RuntimeError("injected ingestion failure")  # noqa: TRY003
        source = await self.registry.resolve(value)
        assert item_id == source.name and definition_name == "pi-event"
        receipt = await self.runtime.ingestion.submit(
            SubmitSourceObservation(
                scope_id=binding.scope_id, observation=project_source_for_transport(self.registry, source)
            )
        )
        if value.event_id == self.fail_event and self.mode == "after":
            raise RuntimeError("failure after durable ingestion")  # noqa: TRY003
        return receipt.source_ref


class RuntimeCheckpoints:
    def __init__(self, runtime):
        self.runtime = runtime

    async def load(self, binding):
        return (await self.runtime.ingestion.checkpoint(binding)).checkpoint

    async def save(self, binding, checkpoint, *, expected):
        await self.runtime.ingestion.commit(
            CommitConnectorCheckpoint(binding=binding, checkpoint=checkpoint, expected=expected)
        )


class Harness:
    def __init__(self, directory):
        self.config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{directory / 'pi.db'}"))
        self.path = directory / "session.jsonl"

    async def binding(self, runtime):
        assert runtime.scopes is not None
        scope = await runtime.scopes.create(ScopeDraft(title="Pi", summary="Offline evidence", idempotency_key="pi"))
        return ConnectorBinding(
            scope_id=scope.scope_id, binding_id="pi", connector_name="pi-session-history", connector_version="1"
        )

    async def run(self, leaf=None, **kwargs):
        async with open_builtin_runtime(self.config) as runtime:
            binding = await self.binding(runtime)
            await runtime.ingestion.register(manifest_for_definition(PI_EVENT_SOURCE_DEFINITION))
            return await ConnectorLifecycle(
                sink=RuntimeSink(runtime, **kwargs), checkpoints=RuntimeCheckpoints(runtime)
            ).run(PiSessionConnector(self.path, leaf), binding)

    async def state(self):
        async with open_builtin_runtime(self.config) as runtime:
            binding = await self.binding(runtime)
            checkpoint = (await runtime.ingestion.checkpoint(binding)).checkpoint
        async with open_builtin_contexts(self.config) as contexts:
            entries = await (await contexts.get(binding.scope_id)).sources.journal.entries()
        return checkpoint, entries


def test_branch_switch_restart_and_off_branch_history_mutation(tmp_path: Path):
    async def scenario():
        harness = Harness(tmp_path)
        records = [
            message("root"),
            message("left", "root", "left evidence"),
            {"type": "model_change", "id": "bridge", "parentId": "root"},
            message("right", "bridge", "right evidence"),
        ]
        harness.path.write_bytes(archive(records))
        left = await harness.run("left")
        right = await harness.run("right")
        assert len(left.items) == 2 and len(right.items) == 1
        checkpoint, entries = await harness.state()
        assert len(entries) == 3
        assert all(isinstance(entry.source, SourceObservation) for entry in entries)
        assert (await harness.run("left")).items == ()
        # Changed raw metadata must fail even on a currently unselected branch.
        records[1]["timestamp"] = "changed"
        harness.path.write_bytes(archive(records))
        with pytest.raises(ValueError, match="rewritten"):
            await harness.run("right")
        assert await harness.state() == (checkpoint, entries)
        del records[1]["timestamp"]
        records.append(message("tail", "right", "appended"))
        harness.path.write_bytes(archive(records) + b"{invalid}\n")
        with pytest.raises(ValueError):
            await harness.run()
        assert await harness.state() == (checkpoint, entries)
        harness.path.write_bytes(archive(records))
        appended = await harness.run()
        assert len(appended.items) == 1 and appended.previous_checkpoint == checkpoint
        assert (await harness.run()).items == ()
        before = await harness.state()
        harness.path.write_bytes(archive(records, session_id="replacement"))
        with pytest.raises(ValueError, match="identity"):
            await harness.run()
        assert await harness.state() == before

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["raise", "reject", "after"])
def test_ingestion_failure_keeps_checkpoint_and_retry_is_durable_and_immutable(tmp_path: Path, mode):
    async def scenario():
        harness = Harness(tmp_path)
        records = [message("root")]
        harness.path.write_bytes(archive(records))
        first = await harness.run()
        records.extend([message("one", "root", "one"), message("two", "one", "two"), message("three", "two", "three")])
        harness.path.write_bytes(archive(records))
        if mode == "reject":
            failed = await harness.run(fail_event="two", mode=mode)
            assert failed.status is ConnectorRunStatus.INCOMPLETE
            assert failed.committed_checkpoint == first.committed_checkpoint
        else:
            with pytest.raises(RuntimeError):
                await harness.run(fail_event="two", mode=mode)
        checkpoint, partial = await harness.state()
        assert checkpoint == first.committed_checkpoint
        assert len(partial) == (3 if mode == "after" else 2)
        # Even uncheckpointed durable events cannot be silently overwritten.
        records[1]["message"]["content"] = "rewrite after interruption"
        harness.path.write_bytes(archive(records))
        with pytest.raises(ValueError, match="conflicts"):
            await harness.run()
        assert await harness.state() == (checkpoint, partial)
        records[1]["message"]["content"] = "one"
        harness.path.write_bytes(archive(records))
        recovered = await harness.run()
        assert recovered.previous_checkpoint == checkpoint
        final_checkpoint, entries = await harness.state()
        assert len(entries) == 4 and final_checkpoint != checkpoint
        assert (await harness.run()).items == ()
        assert await harness.state() == (final_checkpoint, entries)

    asyncio.run(scenario())


def test_every_event_and_large_tool_result_have_exact_durable_citations(tmp_path: Path):
    async def scenario():
        harness = Harness(tmp_path)
        large_text = "toolunique " + "完整原文" * 5000 + " finalsentinel"
        records = [
            message("first", text="firstunique"),
            message("second", "first", "secondunique"),
            message("large", "second", large_text, "toolResult"),
        ]
        records[2]["message"].update(toolName="read", toolCallId="call")
        harness.path.write_bytes(archive(records, bom=True))
        await harness.run()
        _, stored = await harness.state()
        binding = None
        expected = {}
        for entry in stored:
            assert isinstance(entry.source, SourceObservation)
            evidence = TextEvidence.model_validate(entry.source.projection(TEXT_EVIDENCE_PROJECTION_KEY))
            expected[entry.source_ref.source_id] = evidence
        assert max(len(evidence.content.encode()) for evidence in expected.values()) > 8192

        class Pipeline:
            async def extract(self, request, /):
                entries = []
                for source in request.sources:
                    full = TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY)).content
                    for index in range(0, len(full), 1000):
                        entries.append(
                            MemoryEntryInput(kind=f"part-{index}", text=full[index : index + 1000], sources=(source,))
                        )
                return tuple(entries)

        citations = []
        async with open_builtin_runtime(harness.config, candidate_pipeline=Pipeline()) as runtime:
            binding = await harness.binding(runtime)
            memory = runtime.memory.for_scope(binding.scope_id)
            assert (await memory.flush()).source_count == 3
            assert not (await memory.flush()).processed
            for query in ("firstunique", "secondunique", "finalsentinel"):
                hits = (await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=10))).hits
                assert len(hits) == 1
                hit = hits[0]
                citation = MemoryCitation(
                    memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id
                )
                entry = await memory.get(GetMemoryEntryRequest(citation=citation))
                reference = entry.entry.sources[0]
                full = expected[reference.source_id]
                assert query in entry.entry.text and entry.entry.text in full.content
                citations.append((citation, reference.source_id))
        assert len({source_id for _, source_id in citations}) == 3
        async with open_builtin_runtime(harness.config) as runtime:
            memory = runtime.memory.for_scope(binding.scope_id)
            for citation, source_id in citations:
                entry = await memory.get(GetMemoryEntryRequest(citation=citation))
                assert entry.entry.sources[0].source_id == source_id
        async with open_builtin_contexts(harness.config) as contexts:
            context = await contexts.get(binding.scope_id)
            for entry in stored:
                source = await context.sources.get(entry.source)
                assert isinstance(source, SourceObservation)
                evidence = TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY))
                assert evidence == expected[source.name]
                if evidence.metadata["event_id"] == "large":
                    assert evidence.content == "toolResult: [tool_result read call] " + large_text
                    assert source.payload["record"] == records[2]

    asyncio.run(scenario())
