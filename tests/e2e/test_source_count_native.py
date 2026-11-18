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

"""Supplement the published native-worker fixture with accepted-call recovery.

The licensed inputs, local deterministic HTTP provider and original 30/45/5
second bounds are reused unchanged; product code imports none of these helpers.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sqlite3

import pytest

from powercontext.builtin.runtime import open_builtin_runtime

checks = importlib.import_module("competition.challenges.processing-01-count-trigger.acceptance.checks")


@pytest.mark.parametrize("family", ["memory", "topic-memory"])
def test_accepted_provider_failure_retries_after_restart_with_higher_threshold(tmp_path, monkeypatch, family):
    async def scenario():
        database = tmp_path / "accepted-retry.db"
        prefix = "memory" if family == "memory" else "topic_memory"
        binding = f"{family}-source-window"
        with checks.provider(monkeypatch) as server:
            settings = checks.config(
                database,
                server,
                families=(family,),
                **{
                    f"{prefix}_source_count_threshold": 2,
                    "source_window_limit": 1,
                    "topic_memory_source_window_limit": 1,
                },
            )
            server.fail_generation = True
            async with open_builtin_runtime(settings) as runtime:
                sid = await checks.scope(runtime)
                receipts = await checks.capture(runtime, sid, checks.messages()[:2])
                deadline = asyncio.get_running_loop().time() + 30
                while asyncio.get_running_loop().time() < deadline:
                    supervisor = runtime.artifact_processing_supervisor
                    assert supervisor is not None
                    if server.generation_calls and supervisor.family_status[family]["failed"]:
                        break
                    await asyncio.sleep(0.1)
                else:
                    pytest.fail("native provider failure was not observed within 30 seconds")
                await checks.stay_refs(runtime, sid, family, set())
                with sqlite3.connect(database) as connection:
                    row = connection.execute(
                        "SELECT cursor FROM pc_source_cursors WHERE scope_id=? AND binding_name=?", (sid, binding)
                    ).fetchone()
                    assert row is None or json.loads(row[0])["sequence"] == 0
            server.fail_generation = False
            # The accepted durable request must bypass fresh admission, even
            # though the new threshold is now much higher than the pending tail.
            resumed = checks.config(
                database,
                server,
                families=(family,),
                **{
                    f"{prefix}_source_count_threshold": 100,
                    "source_window_limit": 1,
                    "topic_memory_source_window_limit": 1,
                },
            )
            expected = checks.receipt_refs(receipts[:1] if family == "memory" else receipts)
            async with open_builtin_runtime(resumed) as runtime:
                await checks.wait_refs(runtime, sid, family, expected)
                await checks.stay_refs(runtime, sid, family, expected)
                await checks.check_reads(runtime, sid, family, expected)
                with sqlite3.connect(database) as connection:
                    row = connection.execute(
                        "SELECT cursor FROM pc_source_cursors WHERE scope_id=? AND binding_name=?", (sid, binding)
                    ).fetchone()
                    assert row is not None
                    assert json.loads(row[0])["sequence"] == receipts[0 if family == "memory" else 1].sequence
            async with open_builtin_runtime(resumed) as runtime:
                await checks.stay_refs(runtime, sid, family, expected)
                await checks.check_reads(runtime, sid, family, expected)

    asyncio.run(scenario())
