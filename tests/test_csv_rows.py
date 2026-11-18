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

"""CSV edge cases and durable evidence through the public Runtime interfaces."""

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
    GetMemoryEntryRequest,
    SearchMemoryRequest,
    SubmitSourceObservation,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    SourceDefinitionRegistry,
    SourceMaterialization,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)


def parse(content: bytes, *, document_id: str = "table", primary_key: str = "id") -> tuple[CSVRowInput, ...]:
    return parse_csv_document(CSVDocumentInput(document_id=document_id, primary_key=primary_key, content=content))


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\xef\xbb\xbf",
        b"\n",
        b'""\n',
        b"id,id\n",
        b"id, \t\n",
        b"id,,name\n",
        b"id,name\n1,valid\n1,duplicate\n",
        b"id,name\n1,valid\n  ,blank\n",
        b"id,name\n1,valid\n,empty\n",
        b"id,name\n1,valid\n2\n",
        b"id,name\n1,valid\n2,too,wide\n",
        b'id,name\n1,valid\n2,"unterminated',
        b'id,name\n1,valid\n2,"closed"junk\n',
        b"id,name\n1,valid\n2,\xff\n",
        b"id,name\n1,valid\n2,\xc3\n",
        b"id,name\n1,valid\n\xef\xbb\xbf\n",
    ],
)
def test_invalid_document_never_returns_a_valid_prefix(content: bytes) -> None:
    with pytest.raises(ValueError):
        parse(content)


@pytest.mark.parametrize("document_id", ["", " table", "table\t", "\ntable", "table\u00a0"])
def test_document_identity_rejected_even_without_data(document_id: str) -> None:
    with pytest.raises(ValueError):
        parse(b"id,name\n", document_id=document_id)
    with pytest.raises(ValueError):
        csv_row_source_id(document_id, "key")


@pytest.mark.parametrize("primary_key", ["", "missing", " id", "id "])
def test_primary_key_requires_exact_header_match(primary_key: str) -> None:
    with pytest.raises(ValueError):
        parse(b"id,name\n", primary_key=primary_key)


def test_header_and_keys_keep_whitespace_and_blank_physical_records_are_distinct() -> None:
    rows = parse(b' id ,id,note\r\n\r\n key ,001,"line one\r\n\r\nline ""two"""\r\nkey ,002,\r\n', primary_key=" id ")
    assert rows[0].columns == (" id ", "id", "note")
    assert rows[0].values == (" key ", "001", 'line one\r\n\r\nline "two"')
    assert rows[1].values == ("key ", "002", "")
    assert parse(b"id\r\n\r\n\n") == ()
    # Neither a quoted empty field nor delimiter-only records are blank lines.
    for content in (b'id\n""\n', b"id,name\n,\n", b"id\n \n"):
        with pytest.raises(ValueError):
            parse(content)


def test_bom_is_removed_only_at_document_start_and_values_stay_strings() -> None:
    rows = parse("\ufeffid,flag,date,number,empty,note\n001,false,2026-01-02,01.00,,\ufeff文字\n".encode())
    assert rows[0].values == ("001", "false", "2026-01-02", "01.00", "", "\ufeff文字")


def test_component_encoding_collisions_and_unicode_length_bound() -> None:
    assert csv_row_source_id("文:%/", " 值:%/ ") == "csv:%E6%96%87%3A%25%2F:%20%E5%80%BC%3A%25%2F%20"
    assert csv_row_source_id("a:b", "c") != csv_row_source_id("a", "b:c")
    assert csv_row_source_id("%3A", "c") != csv_row_source_id(":", "c")
    assert len(csv_row_source_id("d", "界" * 27 + "x" * 7)) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("d", "界" * 28)
    with pytest.raises(ValueError):
        parse(("id\nok\n" + "界" * 28 + "\n").encode(), document_id="d")
    for key in ("", " ", "\t", "\u00a0"):
        with pytest.raises(ValueError):
            csv_row_source_id("d", key)


def test_direct_row_input_cannot_misrepresent_key_or_drop_fields() -> None:
    row = {
        "document_id": "d",
        "primary_key": "id",
        "primary_key_value": "1",
        "columns": ("id", "note"),
        "values": ("1", ""),
    }
    for update in (
        {"values": ("1",)},
        {"primary_key_value": "other"},
        {"columns": ("id", "id")},
        {"primary_key_value": " ", "values": (" ", "")},
    ):
        with pytest.raises(ValueError):
            CSVRowInput.model_validate(row | update)


def test_unicode_records_survive_ingestion_retry_restart_and_exact_citation(tmp_path: Path) -> None:
    async def scenario() -> None:
        records = parse(
            ' id ,note,empty\n k:%/ ,"uniquetokenalpha 中文",\nk:%/,"uniquetokenbeta café",\n'.encode(),
            document_id="文:%",
            primary_key=" id ",
        )
        reordered = parse(b" id ,note,empty\nk:%/,edited,\n k:%/ ,edited,\n", document_id="文:%", primary_key=" id ")
        registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
        originals = [await registry.resolve(row) for row in records]
        updates = [await registry.resolve(row) for row in reordered]
        assert {source.name for source in originals} == {source.name for source in updates}
        observations = [project_source_for_transport(registry, source) for source in originals]
        expected: dict[str, TextEvidence] = {}
        for source, row, observation in zip(originals, records, observations, strict=True):
            assert isinstance(source, CSVRowSource)
            assert source.materialization is SourceMaterialization.CAPTURED
            assert await registry.read(source) == row
            assert source.name == source.source_id
            expected[source.name] = TextEvidence.model_validate(observation.projection(TEXT_EVIDENCE_PROJECTION_KEY))
            assert json.loads(expected[source.name].content) == dict(zip(row.columns, row.values, strict=True))
            assert list(json.loads(expected[source.name].content)) == list(row.columns)

        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'csv.db'}"))
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="CSV", summary="Unicode evidence", idempotency_key="csv")
            )
            await runtime.ingestion.register(manifest_for_definition(CSV_ROW_SOURCE_DEFINITION))
            for observation in (*observations, observations[0]):
                await runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )

        class Pipeline:
            async def extract(self, request, /) -> tuple[MemoryEntryInput, ...]:
                return tuple(
                    MemoryEntryInput(
                        kind="row", text=source.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"], sources=(source,)
                    )
                    for source in request.sources
                )

        citations = []
        async with open_builtin_runtime(config, candidate_pipeline=Pipeline()) as runtime:
            memory = runtime.memory.for_scope(scope.scope_id)
            assert (await memory.flush()).source_count == len(records)
            assert not (await memory.flush()).processed
            for token, observation in zip(("uniquetokenalpha", "uniquetokenbeta"), observations, strict=True):
                hits = (await memory.search(SearchMemoryRequest(query=token, mode="fts", limit=10))).hits
                assert len(hits) == 1
                hit = hits[0]
                citation = MemoryCitation(
                    memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id
                )
                entry = await memory.get(GetMemoryEntryRequest(citation=citation))
                assert entry.entry.text == expected[observation.name].content
                assert [ref.source_id for ref in entry.entry.sources] == [observation.name]
                citations.append((citation, observation.name))

        async with open_builtin_contexts(config) as contexts:
            entries = await (await contexts.get(scope.scope_id)).sources.journal.entries()
            assert len(entries) == len(records)
            for entry in entries:
                assert isinstance(entry.source, SourceObservation)
                evidence = TextEvidence.model_validate(entry.source.projection(TEXT_EVIDENCE_PROJECTION_KEY))
                assert evidence == expected[entry.source_ref.source_id]

        async with open_builtin_runtime(config) as runtime:
            for citation, source_id in citations:
                entry = await runtime.memory.for_scope(scope.scope_id).get(GetMemoryEntryRequest(citation=citation))
                assert entry.entry.text == expected[source_id].content
                assert entry.entry.sources[0].source_id == source_id

    asyncio.run(scenario())
