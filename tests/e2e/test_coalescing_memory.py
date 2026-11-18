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

"""Check shared provider work through real scoped SQLite Memory operations."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemoryEntryInput
from powercontext.builtin.inference import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts


def test_native_budget_scoped_citations_and_usage(tmp_path):
    async def run():
        texts = tuple(f"Budget example number {index}" for index in range(24))
        calls = []
        attributed = []
        scope = ContextVar("test_scope")

        class Provider:
            profile = EmbeddingProfile(profile_id="scoped-budget", model="offline", dimension=len(texts))

            async def embed(self, batch):
                calls.append(batch)
                return EmbeddingResult(
                    vectors=tuple(tuple(float(text == candidate) for candidate in texts) for text in batch),
                    usage=InferenceUsage(requests=3, input_tokens=7 * len(batch), output_tokens=2),
                )

        model = CoalescingEmbeddingModel(Provider(), max_batch_size=6, coalesce_ms=0, cache_size=48)

        class StartTogether:
            profile = model.profile

            def __init__(self):
                self.arrived = 0
                self.release = asyncio.Event()

            async def embed(self, batch):
                # Match the official barrier at the embedding interface boundary.
                if self.arrived < 6:
                    self.arrived += 1
                    if self.arrived == 6:
                        self.release.set()
                    await self.release.wait()
                result = await model.embed(batch)
                attributed.append((scope.get(), result.usage))
                return result

        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'budget.db'}"))
        async with open_builtin_contexts(config, embedding_model=StartTogether()) as contexts:
            services = [(await contexts.get(f"scope-{index}")).artifacts.memory for index in range(6)]
            entries = tuple(MemoryEntryInput(kind="fact", text=text) for text in texts)

            async def write(index):
                token = scope.set(index)
                try:
                    return await services[index].remember(memory=None, entries=entries, mode="append")
                finally:
                    scope.reset(token)

            memories = await asyncio.wait_for(asyncio.gather(*(write(index) for index in range(6))), 30)
            assert len(calls) == 4
            assert all(len(batch) == 6 for batch in calls)
            assert sorted(text for batch in calls for text in batch) == sorted(texts)
            assert sorted(attributed, key=lambda value: value[0]) == [
                (index, InferenceUsage(requests=12, input_tokens=168, output_tokens=8)) for index in range(6)
            ]
            attributed.clear()
            for index, (service, memory) in enumerate(zip(services, memories, strict=True)):
                assert memory is not None
                stored = await service.entries(memory)
                assert sorted(entry.text for entry in stored) == sorted(texts)
                token = scope.set(index)
                try:
                    result = await service.search(texts[index], memories=(memory,), limit=1, mode="vector")
                finally:
                    scope.reset(token)
                hit = result.hits[0]
                assert hit.text == texts[index]
                assert hit.memory_ref == memory.as_ref()
                exact = next(entry for entry in stored if entry.text == hit.text)
                assert (hit.entry_id, hit.entry_version_id) == (exact.entry_id, exact.entry_version_id)
            assert len(calls) == 4
            assert sorted(attributed, key=lambda value: value[0]) == [
                (index, InferenceUsage(requests=0)) for index in range(6)
            ]
            print("6 scopes, 4 provider batches, 24 unique texts sent once, 6 cache searches, 0 new requests")

    asyncio.run(run())
