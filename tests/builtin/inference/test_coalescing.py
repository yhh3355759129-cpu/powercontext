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

"""Observable coalescing, independent progress, cancellation and provider usage."""

from __future__ import annotations

import asyncio
import gc
import json
from collections import Counter

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemoryEntryInput
from powercontext.builtin.inference import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts


class Metered:
    profile = EmbeddingProfile(profile_id="coalescing-unit", model="offline", dimension=1)

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)
        return EmbeddingResult(
            vectors=tuple((float(len(text)),) for text in texts),
            usage=InferenceUsage(requests=3, input_tokens=11, output_tokens=2),
        )


@pytest.mark.parametrize("window", [0, 10])
@pytest.mark.parametrize("batch_size", [1, 2, 20])
def test_overlapping_calls_preserve_raw_text_order_and_batch_usage(window, batch_size):
    async def scenario():
        base = Metered()
        model = CoalescingEmbeddingModel(base, max_batch_size=batch_size, coalesce_ms=window)
        inputs = ((" X ", "x", " X ", "文字"), ("文字", "x", "X"), ("X", " X "))
        results = await asyncio.gather(*(model.embed(values) for values in inputs))
        assert Counter(text for batch in base.calls for text in batch) == Counter({
            text: 1 for values in inputs for text in values
        })
        assert all(len(batch) <= batch_size for batch in base.calls)
        for values, result in zip(inputs, results, strict=True):
            assert result.vectors == tuple((float(len(text)),) for text in values)
            waited = sum(bool(set(values) & set(batch)) for batch in base.calls)
            assert result.usage == InferenceUsage(
                requests=3 * waited, input_tokens=11 * waited, output_tokens=2 * waited
            )
        cached = await model.embed(tuple(reversed(inputs[0])))
        assert cached.usage.requests == 0 and cached.usage.input_tokens in (None, 0)
        assert model.profile is base.profile

    asyncio.run(scenario())


@pytest.mark.parametrize("window", [0, 10])
def test_late_waiter_shared_usage_and_cancelled_original_with_cache_disabled(window):
    class Blocked(Metered):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts):
            self.calls.append(texts)
            self.started.set()
            await self.release.wait()
            return EmbeddingResult(
                vectors=tuple((float(len(text)),) for text in texts),
                usage=InferenceUsage(requests=7, input_tokens=31, output_tokens=0),
            )

    async def scenario():
        base = Blocked()
        model = CoalescingEmbeddingModel(base, coalesce_ms=window, cache_size=0)
        original = asyncio.create_task(model.embed(("first", "second", "first")))
        await asyncio.wait_for(base.started.wait(), 2)
        late = asyncio.create_task(model.embed(("second", "first", "second")))
        await asyncio.sleep(0)
        original.cancel()
        with pytest.raises(asyncio.CancelledError):
            await original
        base.release.set()
        result = await asyncio.wait_for(late, 2)
        assert result.vectors == ((6.0,), (5.0,), (6.0,))
        assert result.usage == InferenceUsage(requests=7, input_tokens=31, output_tokens=0)
        assert base.calls == [("first", "second")]
        await model.embed(("first", "second"))
        assert base.calls == [("first", "second"), ("first", "second")]

    asyncio.run(scenario())


def test_independent_batches_finish_out_of_order_and_usage_is_not_per_text():
    class OutOfOrder(Metered):
        def __init__(self):
            super().__init__()
            self.first_started = asyncio.Event()
            self.first_release = asyncio.Event()

        async def embed(self, texts):
            self.calls.append(texts)
            if "first" in texts:
                self.first_started.set()
                await self.first_release.wait()
                usage = InferenceUsage(requests=4, input_tokens=None, output_tokens=0)
            else:
                usage = InferenceUsage(requests=9, input_tokens=17, output_tokens=3)
            return EmbeddingResult(vectors=tuple((float(len(text)),) for text in texts), usage=usage)

    async def scenario():
        base = OutOfOrder()
        model = CoalescingEmbeddingModel(base, max_batch_size=2, coalesce_ms=0)
        full = asyncio.create_task(model.embed(("first", "two", "third", "four", "third")))
        await base.first_started.wait()
        independent = await asyncio.wait_for(model.embed(("third", "four")), 2)
        assert independent.vectors == ((5.0,), (4.0,)) and independent.usage.requests == 0
        assert not full.done()  # The independent provider batch did not wait behind the first.
        late = asyncio.create_task(model.embed(("two", "first")))
        await asyncio.sleep(0)
        base.first_release.set()
        result, late_result = await asyncio.gather(full, late)
        assert result.vectors == ((5.0,), (3.0,), (5.0,), (4.0,), (5.0,))
        assert result.usage == InferenceUsage(requests=13, input_tokens=17, output_tokens=3)
        assert late_result.usage == InferenceUsage(requests=4, input_tokens=None, output_tokens=0)
        assert len(base.calls) == 2

    asyncio.run(scenario())


def test_failed_batch_does_not_poison_independent_batch_or_cache():
    class Flaky(Metered):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.fail = True

        async def embed(self, texts):
            if "bad" in texts and self.fail:
                self.calls.append(texts)
                self.started.set()
                await self.release.wait()
                raise RuntimeError("injected batch failure")  # noqa: TRY003
            return await super().embed(texts)

    async def scenario():
        base = Flaky()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
        failed = asyncio.create_task(model.embed(("bad", "good")))
        shared = asyncio.create_task(model.embed(("bad",)))
        await base.started.wait()
        good = await asyncio.wait_for(model.embed(("good",)), 2)
        assert good.vectors == ((4.0,),) and good.usage.requests == 0
        base.release.set()
        results = await asyncio.gather(failed, shared, return_exceptions=True)
        assert all(isinstance(result, RuntimeError) for result in results)
        base.fail = False
        repaired = await model.embed(("good", "bad"))
        assert repaired.usage == InferenceUsage(requests=3, input_tokens=11, output_tokens=2)
        assert Counter(text for batch in base.calls for text in batch) == Counter({"bad": 2, "good": 1})

    asyncio.run(scenario())


@pytest.mark.parametrize("count", [0, 1, 3])
def test_wrong_vector_count_fails_entire_batch_and_is_retryable(count):
    class Wrong(Metered):
        async def embed(self, texts):
            self.calls.append(texts)
            return EmbeddingResult(vectors=tuple((1.0,) for _ in range(count if len(self.calls) == 1 else len(texts))))

    async def scenario():
        base = Wrong()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        with pytest.raises(ValueError):
            await model.embed(("one", "two"))
        assert (await model.embed(("one", "two"))).vectors == ((1.0,), (1.0,))
        assert len(base.calls) == 2

    asyncio.run(scenario())


def test_lru_access_refresh_empty_mixed_hits_and_return_value_isolation():
    async def scenario():
        base = Metered()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=2)
        assert (await model.embed(())).usage.requests == 0 and base.calls == []
        initial = await model.embed(("a", "bb"))
        initial.vectors = ((999.0,),)
        initial.usage.requests = 999
        assert (await model.embed(("a",))).vectors == ((1.0,),)
        assert (await model.embed(("a", "bb", "a"))).usage.requests == 0
        mixed = await model.embed(("a", "ccc", "a"))
        assert mixed.vectors == ((1.0,), (3.0,), (1.0,)) and mixed.usage.requests == 3
        assert (await model.embed(("a",))).usage.requests == 0
        await model.embed(("bb",))
        assert base.calls == [("a", "bb"), ("ccc",), ("bb",)]

    asyncio.run(scenario())


def test_wrappers_and_provider_owned_result_do_not_share_cache_state():
    class Retained(Metered):
        async def embed(self, texts):
            result = await super().embed(texts)
            self.retained = result
            return result

    async def scenario():
        base = Retained()
        first = CoalescingEmbeddingModel(base, coalesce_ms=0)
        second = CoalescingEmbeddingModel(base, coalesce_ms=0)
        await first.embed(("same",))
        base.retained.vectors = ((99.0,),)
        assert (await first.embed(("same",))).vectors == ((4.0,),)
        assert (await second.embed(("same",))).usage.requests == 3
        assert len(base.calls) == 2

    asyncio.run(scenario())


def test_abandoned_failure_has_no_unretrieved_exceptions_or_stuck_retry():
    class Failing(Metered):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.done = asyncio.Event()
            self.fail = True

        async def embed(self, texts):
            if not self.fail:
                return await super().embed(texts)
            self.started.set()
            await self.release.wait()
            self.done.set()
            raise RuntimeError("abandoned provider failure")  # noqa: TRY003

    async def scenario():
        loop = asyncio.get_running_loop()
        errors = []
        old = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        try:
            base = Failing()
            model = CoalescingEmbeddingModel(base, coalesce_ms=0)
            callers = [asyncio.create_task(model.embed(("same",))) for _ in range(2)]
            await base.started.wait()
            for task in callers:
                task.cancel()
            assert all(
                isinstance(value, asyncio.CancelledError)
                for value in await asyncio.gather(*callers, return_exceptions=True)
            )
            base.release.set()
            await base.done.wait()
            await asyncio.sleep(0)
            gc.collect()
            assert errors == []
            base.fail = False
            assert (await model.embed(("same",))).vectors == ((4.0,),)
        finally:
            loop.set_exception_handler(old)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "settings",
    [
        {"max_batch_size": True},
        {"max_batch_size": 0},
        {"max_batch_size": 1.5},
        {"cache_size": True},
        {"cache_size": -1},
        {"coalesce_ms": True},
        {"coalesce_ms": -1},
        {"coalesce_ms": float("nan")},
        {"coalesce_ms": float("inf")},
        {"coalesce_ms": "5"},
    ],
)
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(Metered(), **settings)


def test_native_six_scope_budget_exact_citations_and_non_unit_provider_usage(tmp_path):
    async def scenario():
        texts = tuple(f"axis{index} fixture" for index in range(24))

        class Native:
            profile = EmbeddingProfile(profile_id="coalesced-native", model="offline", dimension=len(texts))

            def __init__(self):
                self.calls = []

            async def embed(self, values):
                self.calls.append(values)
                return EmbeddingResult(
                    vectors=tuple(tuple(float(value == text) for text in texts) for value in values),
                    usage=InferenceUsage(requests=3, input_tokens=len(values) * 7, output_tokens=2),
                )

        base = Native()
        model = CoalescingEmbeddingModel(base, max_batch_size=6, coalesce_ms=0, cache_size=48)

        class Gate:
            profile = model.profile

            def __init__(self):
                self.arrivals = 0
                self.release = asyncio.Event()
                self.results = []

            async def embed(self, values):
                if self.arrivals < 6:
                    self.arrivals += 1
                    if self.arrivals == 6:
                        self.release.set()
                    await self.release.wait()
                result = await model.embed(values)
                self.results.append(result)
                return result

        gate = Gate()
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'native.db'}"))
        async with open_builtin_contexts(config, embedding_model=gate) as contexts:
            services = [(await contexts.get(f"scope-{index}")).artifacts.memory for index in range(6)]
            entries = tuple(MemoryEntryInput(kind="fact", text=text) for text in texts)
            memories = await asyncio.wait_for(
                asyncio.gather(
                    *(service.remember(memory=None, entries=entries, mode="append") for service in services)
                ),
                30,
            )
            assert len(base.calls) == 4
            assert [len(batch) for batch in base.calls] == [6, 6, 6, 6]
            assert Counter(value for batch in base.calls for value in batch) == Counter(texts)
            assert len(gate.results) == 6
            assert all(
                result.usage == InferenceUsage(requests=12, input_tokens=168, output_tokens=8)
                for result in gate.results
            )
            for index, (service, memory) in enumerate(zip(services, memories, strict=True)):
                assert memory is not None
                result = await service.search(texts[index], memories=(memory,), limit=1, mode="vector")
                hit = result.hits[0]
                assert hit.text == texts[index] and hit.memory_ref == memory.as_ref()
                expanded = await service.expand(result.hits)
                assert len(expanded) == 1
                assert expanded[0].entry_id == hit.entry_id and expanded[0].entry_version_id == hit.entry_version_id
                assert expanded[0].text == hit.text
            assert len(base.calls) == 4
            assert all(result.usage.requests == 0 for result in gate.results[6:])
            print(
                json.dumps(
                    {
                        "scopes": 6,
                        "distinct_texts": 24,
                        "provider_batches": len(base.calls),
                        "batch_sizes": [len(batch) for batch in base.calls],
                        "cache_query_extra_batches": len(base.calls) - 4,
                        "write_usage_per_scope": gate.results[0].usage.model_dump(),
                    },
                    sort_keys=True,
                )
            )

    asyncio.run(scenario())
