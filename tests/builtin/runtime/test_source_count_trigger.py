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

"""Count admission uses actual durable identities and independent success cursors."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import insert

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.sqlite import SQLiteConfig, SQLiteProfile
from powercontext.builtin.persistence.tables import SHARED_TABLES, SOURCES_TABLE
from powercontext.builtin.runtime import BuiltinConfig, RuntimeConfig
from powercontext.builtin.runtime.processing_discovery import source_count_scopes
from powercontext.builtin.sources import SourceCursor
from powercontext.server.settings import ServerSettings


@pytest.mark.parametrize("field", ["memory_source_count_threshold", "topic_memory_source_count_threshold"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 2.5, "3", "", [], {}])
def test_public_configuration_rejects_every_non_positive_integer_form(field, value):
    for model, payload in (
        (RuntimeConfig, {field: value}),
        (BuiltinConfig, {"runtime": {field: value}}),
        (ServerSettings, {"runtime": {field: value}}),
    ):
        with pytest.raises(ValidationError):
            model.model_validate(payload)
        with pytest.raises(ValidationError):
            model.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("value", [None, 1, 3, 100])
def test_public_configuration_preserves_defaults_and_independent_thresholds(value):
    settings = ServerSettings.model_validate({
        "runtime": {"memory_source_count_threshold": value, "topic_memory_source_count_threshold": 7}
    })
    assert settings.runtime.memory_source_count_threshold == value
    assert settings.runtime.topic_memory_source_count_threshold == 7
    assert RuntimeConfig().memory_source_count_threshold is None
    assert RuntimeConfig().topic_memory_source_count_threshold is None


def test_sparse_journal_positions_count_identities_and_reopen_with_family_cursors(tmp_path: Path):
    config = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'journal.db'}")

    async def seed():
        async with (
            SQLiteProfile.open(config, tables=SHARED_TABLES) as profile,
            profile.database.transaction() as connection,
        ):
            # Synthetic metadata positions deliberately contain holes. The
            # admission query must neither decode bodies nor subtract heads.
            for scope, positions in (("left", (3, 40, 900)), ("right", (10, 20))):
                for position in positions:
                    await connection.execute(
                        insert(SOURCES_TABLE).values(
                            scope_id=scope,
                            source_type="content",
                            source_id=f"row-{position}",
                            payload=b"not-a-source-payload",
                            journal_position=position,
                        )
                    )
            assert await source_count_scopes(connection, ("left", "right"), "memory-source-window", 3) == frozenset({
                "left"
            })
            assert await source_count_scopes(connection, ("right",), "memory-source-window", 3) == frozenset()
            await SourceCursorRepository().save(
                connection, "left", "memory-source-window", SourceCursor(sequence=3), expected_generation=None
            )
            assert await source_count_scopes(connection, ("left", "right"), "memory-source-window", 3) == frozenset()
            # Topic has not consumed Memory's three identities.
            assert await source_count_scopes(
                connection, ("left", "right"), "topic-memory-source-window", 3
            ) == frozenset({"left"})

    async def reopen():
        async with (
            SQLiteProfile.open(config, tables=SHARED_TABLES) as profile,
            profile.database.transaction() as connection,
        ):
            assert await source_count_scopes(connection, ("left", "right"), "memory-source-window", 2) == frozenset({
                "left",
                "right",
            })
            assert await source_count_scopes(connection, ("left",), "memory-source-window", 3) == frozenset()
            assert await source_count_scopes(connection, ("left",), "topic-memory-source-window", 3) == frozenset({
                "left"
            })
            assert await source_count_scopes(connection, (), "memory-source-window", 1) == frozenset()
            assert await source_count_scopes(connection, ("left",), "memory-source-window", 10**30) == frozenset()

    asyncio.run(seed())
    asyncio.run(reopen())
