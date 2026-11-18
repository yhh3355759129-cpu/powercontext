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

"""Durable evidence DAG propagation with replay-safe native Memory transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.source_lifecycle import load_nodes, save_node
from powercontext.builtin.persistence.sources import SourceRepository
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import BUILTIN_SOURCE_REGISTRY
from powercontext.builtin.sources.journal import validate_scope_id
from powercontext.sources import SourceMaterialization, SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """Invalid evidence binding, event, or generation use."""


class SourceLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(strict=True)
    revision: int = Field(strict=True, gt=0)
    sequence: int = Field(strict=True, gt=0)
    available: StrictBool
    reason: str = Field(strict=True)

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle text must not be blank")  # noqa: TRY003
        return value


class _Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    kind: Literal["source", "memory", "topic"]
    reference: dict[str, JsonValue]
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    revision: int | None = None
    available: bool = True
    event: SourceLifecycleEvent | None = None
    complete: bool = True

    def binding(self) -> dict[str, object]:
        return self.model_dump(exclude={"available", "event", "complete"})


class SourceLifecycleApplication:
    def __init__(
        self, runtime: BuiltinRuntime, database: AsyncDatabase | None, sources: SourceRepository | None = None
    ) -> None:
        self._runtime = runtime
        self._database = database
        self._sources = sources if sources is not None else SourceRepository(BUILTIN_SOURCE_REGISTRY)

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        if self._database is None:
            raise SourceLifecycleError("lifecycle requires a relational runtime")  # noqa: TRY003
        return ScopedSourceLifecycle(self._runtime, self._database, self._sources, validate_scope_id(scope_id))


class ScopedSourceLifecycle:
    def __init__(
        self, runtime: BuiltinRuntime, database: AsyncDatabase, sources: SourceRepository, scope_id: str
    ) -> None:
        self._runtime, self._database, self.scope_id = runtime, database, scope_id
        self._sources = sources

    async def _load(self) -> dict[str, _Node]:
        async with self._database.transaction() as connection:
            rows = await load_nodes(connection, self.scope_id)
        nodes = tuple(_Node.model_validate_json(row["payload"]) for row in rows)
        return {node.node_id: node for node in nodes}

    async def _save(self, nodes: tuple[_Node, ...], *, new_ordinal: int | None = None) -> None:
        async with self._database.transaction() as connection:
            for node in nodes:
                artifact_id = ArtifactRef.model_validate(node.reference).artifact_id if node.kind == "topic" else None
                await save_node(
                    connection,
                    self.scope_id,
                    node.node_id,
                    payload=node.model_dump_json().encode(),
                    available=node.available,
                    kind=node.kind,
                    artifact_id=artifact_id,
                    ordinal=new_ordinal,
                )

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        if type(revision) is not int or revision < 1 or not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("Source registration requires a SourceRef and positive revision")  # noqa: TRY003
        await self._register(
            _Node(
                node_id=self._name(node_id),
                kind="source",
                reference=source_ref.model_dump(mode="json"),
                revision=revision,
            )
        )

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: tuple[str, ...], policy: str
    ) -> None:
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("Memory registration requires an exact MemoryCitation")  # noqa: TRY003
        await self._derived(node_id, "memory", citation.model_dump(mode="json"), parents, policy)

    async def register_topic(
        self, node_id: str, artifact_ref: ArtifactRef, *, parents: tuple[str, ...], policy: str
    ) -> None:
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("Topic registration requires a Topic Memory ArtifactRef")  # noqa: TRY003
        await self._derived(node_id, "topic", artifact_ref.model_dump(mode="json"), parents, policy)

    async def _derived(self, node_id, kind, reference, parents, policy):
        self._name(node_id)
        if (
            type(parents) is not tuple
            or not parents
            or any(not isinstance(parent, str) or not parent.strip() for parent in parents)
            or len(set(parents)) != len(parents)
            or node_id in parents
            or policy not in ("all", "any")
        ):
            raise SourceLifecycleError("derived nodes require unique existing parents and all/any policy")  # noqa: TRY003
        await self._register(_Node(node_id=node_id, kind=kind, reference=reference, parents=parents, policy=policy))

    @staticmethod
    def _name(node_id: str) -> str:
        if not isinstance(node_id, str) or not node_id.strip():
            raise SourceLifecycleError("node_id must be a non-blank string")  # noqa: TRY003
        return node_id

    async def _validate_captured_source(self, ref: SourceRef) -> None:
        async with self._database.transaction() as connection:
            stored = await self._sources.get(connection, self.scope_id, ref)
        if stored.value.materialization != SourceMaterialization.CAPTURED:
            raise SourceLifecycleError("lifecycle requires a captured Source")  # noqa: TRY003

    async def _validate_reference(self, node: _Node) -> None:
        try:
            if node.kind == "source":
                await self._validate_captured_source(SourceRef.model_validate(node.reference))
            elif node.kind == "memory":
                await self._runtime.memory.for_scope(self.scope_id).get(
                    GetMemoryEntryRequest(citation=MemoryCitation.model_validate(node.reference))
                )
            else:
                ref = ArtifactRef.model_validate(node.reference)
                await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                    ref.family, ref.artifact_id, ref.revision
                )
        except SourceLifecycleError:
            raise
        except Exception as error:
            raise SourceLifecycleError("registered reference must be readable in this scope") from error  # noqa: TRY003

    async def _register(self, node: _Node) -> None:
        async with self._runtime._scope_operation(self.scope_id):
            nodes = await self._load()
            existing = nodes.get(node.node_id)
            if existing is not None:
                if existing.binding() != node.binding():
                    raise SourceLifecycleError("node is already bound differently")  # noqa: TRY003
            else:
                if any(parent not in nodes for parent in node.parents):
                    raise SourceLifecycleError("parents must be registered before their children")  # noqa: TRY003
                await self._validate_reference(node)
                if node.kind != "source":
                    supported = [nodes[parent].available for parent in node.parents]
                    node.available = all(supported) if node.policy == "all" else any(supported)
                node.complete = False
                await self._save((node,), new_ordinal=len(nodes))
                nodes[node.node_id] = node
            await self._synchronize(nodes, "source lifecycle registration")

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("apply requires SourceLifecycleEvent")  # noqa: TRY003
        async with self._runtime._scope_operation(self.scope_id):
            nodes = await self._load()
            root = nodes.get(event.node_id)
            if root is None or root.kind != "source" or root.revision != event.revision:
                raise SourceLifecycleError("event requires a registered Source and matching revision")  # noqa: TRY003
            if root.event is not None:
                if event.sequence < root.event.sequence or (
                    event.sequence == root.event.sequence and event != root.event
                ):
                    raise SourceLifecycleError("event is stale or conflicts with an accepted sequence")  # noqa: TRY003
                if event == root.event and all(node.complete for node in nodes.values()):
                    return sorted(node.node_id for node in nodes.values() if node.available)
            root.available = event.available
            root.event = event
            root.complete = False
            for node in nodes.values():
                if node.kind != "source":
                    supported = [nodes[parent].available for parent in node.parents]
                    node.available = all(supported) if node.policy == "all" else any(supported)
            # Accept the intent and desired graph atomically, but do not mark
            # the event complete before every native write has succeeded.
            await self._save(tuple(nodes.values()))
            await self._synchronize(nodes, event.reason)
            return sorted(node.node_id for node in nodes.values() if node.available)

    async def _synchronize(self, nodes: dict[str, _Node], reason: str) -> None:
        # Aliases for one original body are supported if any binding is active.
        memories: dict[tuple[str, str, str], tuple[MemoryCitation, bool]] = {}
        for node in nodes.values():
            if node.kind == "memory":
                citation = MemoryCitation.model_validate(node.reference)
                key = (citation.memory_ref.artifact_id, citation.entry_id, citation.entry_version_id)
                previous = memories.get(key)
                memories[key] = (citation, node.available or (previous is not None and previous[1]))
        for citation, available in memories.values():
            async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
                service = context.artifacts.memory
                current = await service.head(citation.memory_ref.artifact_id)
                entries = await service.entries(current)
                entry = next(
                    (
                        entry
                        for entry in entries
                        if entry.entry_id == citation.entry_id and entry.entry_version_id == citation.entry_version_id
                    ),
                    None,
                )
                if entry is None:
                    raise SourceLifecycleError("registered Memory body is no longer current")  # noqa: TRY003
                state = next(item.state for item in current.content.manifest.entries if item.entry_id == entry.entry_id)
                if (state == "active") != available:
                    operation = service.reactivate if available else service.forget
                    await operation(current, entries=(entry,), reason=reason)
        pending = tuple(node for node in nodes.values() if not node.complete)
        for node in pending:
            node.complete = True
        if pending:
            await self._save(pending)

    async def resolve(self, node_id: str, *, purpose: str):
        self._name(node_id)
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError("unknown lifecycle purpose")  # noqa: TRY003
        async with self._runtime._scope_operation(self.scope_id):
            node = (await self._load()).get(node_id)
            if node is None or (purpose == "generation" and not node.available):
                raise SourceLifecycleError("unknown or unavailable lifecycle node")  # noqa: TRY003
            model = {"source": SourceRef, "memory": MemoryCitation, "topic": ArtifactRef}[node.kind]
            return model.model_validate(node.reference)


__all__ = ["SourceLifecycleError", "SourceLifecycleEvent"]
