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

"""Lifecycle validation, scoped real references and interrupted native commits."""

from __future__ import annotations

import asyncio
import importlib
import sqlite3

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    GetTopicMemoryRequest,
    RememberMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError, SourceLifecycleEvent
from powercontext.sources import SourceRef

checks = importlib.import_module("competition.challenges.memory-01-source-lifecycle.acceptance.checks")


def event_payload():
    return {"node_id": "source", "revision": 3, "sequence": 1, "available": False, "reason": "withdrawal"}


@pytest.mark.parametrize("field", ["revision", "sequence"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 1.5, "1", None])
def test_event_counters_are_strict_positive_integers(field, value):
    with pytest.raises(ValidationError):
        SourceLifecycleEvent.model_validate(event_payload() | {field: value})


@pytest.mark.parametrize("field", ["node_id", "reason"])
@pytest.mark.parametrize("value", ["", " \t\n", True, 1, b"text", None, []])
def test_event_names_are_nonblank_strict_strings(field, value):
    with pytest.raises(ValidationError):
        SourceLifecycleEvent.model_validate(event_payload() | {field: value})


@pytest.mark.parametrize("value", [0, 1, "true", "false", None, []])
def test_availability_is_strict_boolean_and_extra_fields_are_forbidden(value):
    with pytest.raises(ValidationError):
        SourceLifecycleEvent.model_validate(event_payload() | {"available": value})
    with pytest.raises(ValidationError):
        SourceLifecycleEvent.model_validate(event_payload() | {"extra": True})


def test_registration_replay_conflicts_missing_and_cross_scope_references():
    async def scenario():
        async with open_builtin_runtime(BuiltinConfig()) as runtime:
            left, app, bindings, refs = await checks.build(runtime, suffix="left")
            right, other, _, other_refs = await checks.build(runtime, suffix="right")
            assert left != right
            before = (await runtime.memory.for_scope(left).list()).memory_ref
            await app.register_source("root0", refs["root0"], revision=3)
            await app.register_memory(
                "needle0", refs["needle0"], parents=bindings["needle0"][0], policy=bindings["needle0"][1]
            )
            await app.register_topic(
                "topicneedle0", refs["topicneedle0"], parents=bindings["topicneedle0"][0], policy="all"
            )
            assert (await runtime.memory.for_scope(left).list()).memory_ref == before
            for revision in (True, False, 0, 1.5, "3"):
                with pytest.raises(SourceLifecycleError):
                    await app.register_source("bad", refs["root0"], revision=revision)
            with pytest.raises(SourceLifecycleError):
                await app.register_source("root0", refs["root0"], revision=4)
            with pytest.raises(SourceLifecycleError):
                await app.register_source("root0", refs["root1"], revision=3)
            with pytest.raises(SourceLifecycleError):
                await app.register_memory("root0", refs["needle0"], parents=("root1",), policy="all")
            for parents, policy in (
                ((), "all"),
                (("missing",), "all"),
                (("root0", "root0"), "any"),
                (("bad",), "all"),
                (("root0",), "invalid"),
            ):
                with pytest.raises(SourceLifecycleError):
                    await app.register_memory("bad", refs["needle0"], parents=parents, policy=policy)
            with pytest.raises(SourceLifecycleError):
                await app.register_source("missing", SourceRef(source_type="content", source_id="absent"), revision=3)
            with pytest.raises(SourceLifecycleError):
                await app.register_memory("foreign-memory", other_refs["needle0"], parents=("root0",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await app.register_topic("foreign-topic", other_refs["topicneedle0"], parents=("root0",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await app.register_topic(
                    "missing-topic",
                    ArtifactRef(family="topic-memory", artifact_id="absent", revision=1),
                    parents=("root0",),
                    policy="all",
                )
            withdrawn = checks.event("root0", 1, False)
            await app.apply(withdrawn)
            assert await other.resolve("needle0", purpose="generation") == other_refs["needle0"]
            for invalid in (
                checks.event("needle0", 1, True),
                checks.event("absent", 1, True),
                checks.event("root0", 1, True),
                withdrawn.model_copy(update={"reason": "conflict"}),
                withdrawn.model_copy(update={"revision": 4}),
            ):
                with pytest.raises(SourceLifecycleError):
                    await app.apply(invalid)
            for name, purpose in (("absent", "audit"), ("root0", "unknown"), ("needle0", "generation")):
                with pytest.raises(SourceLifecycleError):
                    await app.resolve(name, purpose=purpose)
            assert await app.resolve("needle0", purpose="audit") == refs["needle0"]
            snapshot = (await runtime.memory.for_scope(left).list(include_inactive=True)).memory_ref
            await app.apply(withdrawn)
            assert (await runtime.memory.for_scope(left).list(include_inactive=True)).memory_ref == snapshot

    asyncio.run(scenario())


@pytest.mark.parametrize("restart", [False, True])
def test_real_memory_commit_interrupts_mid_propagation_and_replay_completes_once(tmp_path, restart):
    async def scenario():
        database = tmp_path / "failure.db"
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{database}"))
        async with open_builtin_runtime(config) as runtime:
            sid, app, bindings, refs = await checks.build(runtime)
            initial = (await runtime.memory.for_scope(sid).list()).memory_ref
            assert initial is not None
            with sqlite3.connect(database) as connection:
                # First native state transition commits; the next Memory
                # revision insert fails inside its actual SQL transaction.
                connection.execute(
                    "CREATE TRIGGER lifecycle_interrupt BEFORE INSERT ON pc_artifacts "
                    f"WHEN NEW.family='memory' AND NEW.revision>={initial.revision + 2} "
                    "BEGIN SELECT RAISE(ABORT, 'lifecycle controlled commit failure'); END"
                )
            withdrawal = checks.event("root0", 1, False)
            with pytest.raises(IntegrityError, match="lifecycle controlled commit failure"):
                await app.apply(withdrawal)
            partial = await runtime.memory.for_scope(sid).list(include_inactive=True)
            assert partial.memory_ref is not None
            assert partial.memory_ref.revision == initial.revision + 1
            assert sum(row.state != "active" for row in partial.entries) == 1
            for node in bindings:
                if bindings[node][2] == "memory":
                    original = await runtime.memory.for_scope(sid).get(GetMemoryEntryRequest(citation=refs[node]))
                    assert original.entry.text == f"{node} fixture evidence"
            with pytest.raises(SourceLifecycleError):
                await app.apply(withdrawal.model_copy(update={"reason": "not the accepted event"}))
            assert (await runtime.memory.for_scope(sid).list(include_inactive=True)).memory_ref == partial.memory_ref
            with sqlite3.connect(database) as connection:
                connection.execute("DROP TRIGGER lifecycle_interrupt")
            if not restart:
                await app.apply(withdrawal)
                await checks.assert_runtime(runtime, sid, bindings, refs, checks.oracle(bindings, {"root0"}))
        async with open_builtin_runtime(config) as runtime:
            app = runtime.lifecycle.for_scope(sid)
            expected = checks.oracle(bindings, {"root0"})
            assert await app.apply(withdrawal) == sorted(expected)
            await checks.assert_runtime(runtime, sid, bindings, refs, expected)
            finished = (await runtime.memory.for_scope(sid).list(include_inactive=True)).memory_ref
            assert finished is not None
            affected = sum(kind == "memory" and node not in expected for node, (_, _, kind) in bindings.items())
            assert finished.revision == initial.revision + affected
            await app.apply(withdrawal)
            assert (await runtime.memory.for_scope(sid).list(include_inactive=True)).memory_ref == finished
            await app.apply(checks.event("root0", 2, True))
            await checks.assert_runtime(runtime, sid, bindings, refs, checks.oracle(bindings, set()))
            restored = (await runtime.memory.for_scope(sid).list()).memory_ref
            await app.apply(checks.event("root0", 2, True))
            assert (await runtime.memory.for_scope(sid).list()).memory_ref == restored
            for node, ref in refs.items():
                assert await app.resolve(node, purpose="audit") == ref
                assert await app.resolve(node, purpose="generation") == ref

    asyncio.run(scenario())


def test_registration_failure_after_withdrawal_is_repaired_by_same_event(tmp_path):
    async def scenario():
        database = tmp_path / "registration.db"
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{database}"))
        async with open_builtin_runtime(config) as runtime:
            sid, app, _, _ = await checks.build(runtime)
            withdrawal = checks.event("root0", 1, False)
            await app.apply(withdrawal)
            created = await runtime.memory.for_scope(sid).remember(
                RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text="Late binding original body."),))
            )
            assert created.entry is not None
            ref = created.entry.citation
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TRIGGER registration_interrupt BEFORE INSERT ON pc_artifacts WHEN NEW.family='memory' BEGIN SELECT RAISE(ABORT,'registration interrupted'); END"
                )
            with pytest.raises(IntegrityError, match="registration interrupted"):
                await app.register_memory("late", ref, parents=("root0",), policy="all")
            with sqlite3.connect(database) as connection:
                connection.execute("DROP TRIGGER registration_interrupt")
        async with open_builtin_runtime(config) as runtime:
            app = runtime.lifecycle.for_scope(sid)
            assert "late" not in await app.apply(withdrawal)
            rows = await runtime.memory.for_scope(sid).list(include_inactive=True)
            assert next(row.state for row in rows.entries if row.entry.entry_id == ref.entry_id) == "inactive"
            assert await app.resolve("late", purpose="audit") == ref
            original = await runtime.memory.for_scope(sid).get(GetMemoryEntryRequest(citation=ref))
            assert original.entry.text == "Late binding original body."

    asyncio.run(scenario())


def test_topic_browse_hides_withdrawn_heads_but_exact_get_keeps_original_body():
    async def scenario():
        async with open_builtin_runtime(BuiltinConfig()) as runtime:
            sid, app, bindings, refs = await checks.build(runtime)
            topics = runtime.topic_memory.for_scope(sid)
            for available, sequence in ((False, 1), (True, 2)):
                await app.apply(checks.event("root0", sequence, available))
                expected = checks.oracle(bindings, set() if available else {"root0"})
                visible = await topics.browse(limit=100)
                assert {item.title for item in visible} == {name for name in expected if name.startswith("topic")}
                for node, (_, _, kind) in bindings.items():
                    if kind == "topic":
                        published = await topics.get(GetTopicMemoryRequest(artifact=refs[node]))
                        assert published.topic.as_ref() == refs[node]
                        assert published.topic.content.title == node

    asyncio.run(scenario())


def test_register_captured_observation_after_restart(tmp_path):
    from powercontext.builtin.runtime import SubmitSourceObservation
    from powercontext.builtin.scope import ScopeDraft
    from powercontext.sources import SourceDefinitionRegistry, manifest_for_definition, project_source_for_transport
    from powercontext.sources.csv_rows import CSV_ROW_SOURCE_DEFINITION, CSVDocumentInput, parse_csv_document

    async def scenario():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'rows.db'}"))
        registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
        (row,) = parse_csv_document(
            CSVDocumentInput(document_id="receipts", primary_key="id", content=b"id,total\n1,27\n")
        )
        observation = project_source_for_transport(registry, await registry.resolve(row))
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(ScopeDraft(title="Rows", summary="Receipts", idempotency_key="rows"))
            await runtime.ingestion.register(manifest_for_definition(CSV_ROW_SOURCE_DEFINITION))
            receipt = await runtime.ingestion.submit(
                SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
            )
        async with open_builtin_runtime(config) as runtime:
            service = runtime.lifecycle.for_scope(scope.scope_id)
            await service.register_source("row", receipt.source_ref, revision=3)
            assert await service.apply(checks.event("row", 1, False)) == []
            assert await service.resolve("row", purpose="audit") == receipt.source_ref

    asyncio.run(scenario())
