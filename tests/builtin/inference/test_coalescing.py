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

from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel


class Provider:
    profile = EmbeddingProfile(profile_id="coalescing-test", model="offline", dimension=1)

    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.fail = False
        self.extra = 0

    async def embed(self, texts):
        self.calls.append(texts)
        self.started.set()
        await self.release.wait()
        if self.fail and "bad" in texts:
            raise RuntimeError("provider-failure")
        vectors = tuple((float(sum(map(ord, text))),) for text in texts)
        if self.extra < 0:
            vectors = vectors[:-1]
        elif self.extra > 0:
            vectors += ((99.0,),)
        return EmbeddingResult(
            vectors=vectors,
            usage=InferenceUsage(requests=3, input_tokens=7 * len(texts), output_tokens=2),
        )


@pytest.mark.parametrize("window", [0, 5])
def test_overlap_late_waiter_usage_and_caller_mutation(window):
    async def run():
        base = Provider()
        base.release.clear()
        model = CoalescingEmbeddingModel(base, coalesce_ms=window, max_batch_size=2)
        assert model.profile is base.profile
        first = asyncio.create_task(model.embed(("a", "b", "a")))
        await base.started.wait()
        late = asyncio.create_task(model.embed(("b", "a", "b")))
        await asyncio.sleep(0)
        base.release.set()
        left, right = await asyncio.gather(first, late)
        assert left.vectors == ((97.0,), (98.0,), (97.0,))
        assert right.vectors == ((98.0,), (97.0,), (98.0,))
        assert left.usage == right.usage == InferenceUsage(requests=3, input_tokens=14, output_tokens=2)
        assert base.calls == [("a", "b")]
        left.vectors = ((0.0,),)
        left.usage.requests = 900
        cached = await model.embed(("a", "b"))
        assert cached.vectors == ((97.0,), (98.0,))
        assert cached.usage.requests == 0
        mixed = await model.embed(("a", "c", "c"))
        assert mixed.usage == InferenceUsage(requests=3, input_tokens=7, output_tokens=2)
        assert mixed.vectors == ((97.0,), (99.0,), (99.0,))
        assert (await model.embed(())).usage.requests == 0
        assert len(base.calls) == 2

    asyncio.run(run())


def test_out_of_order_batches_and_independent_failure():
    class Reordered(Provider):
        async def embed(self, texts):
            if "bad" in texts:
                return await super().embed(texts)
            self.calls.append(texts)
            self.started.set()
            return EmbeddingResult(vectors=((100.0,),), usage=InferenceUsage(requests=4, input_tokens=8))

    async def run():
        base = Reordered()
        base.fail = True
        base.release.clear()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, max_batch_size=1)
        combined = asyncio.create_task(model.embed(("bad", "good", "good")))
        good = asyncio.create_task(model.embed(("good",)))
        assert (await asyncio.wait_for(good, 2)).usage.requests == 4
        assert not combined.done()
        base.release.set()
        with pytest.raises(RuntimeError, match="provider-failure"):
            await combined
        base.fail = False
        recovered = await model.embed(("bad", "good"))
        assert recovered.usage.requests == 3
        assert base.calls.count(("bad",)) == 2
        assert base.calls.count(("good",)) == 1

    asyncio.run(run())


def test_successful_batches_finish_in_reverse_and_sum_provider_usage():
    class Reverse(Provider):
        async def embed(self, texts):
            if "a" in texts:
                await self.release.wait()
            else:
                self.started.set()
            return EmbeddingResult(
                vectors=tuple((float(ord(t)),) for t in texts),
                usage=InferenceUsage(requests=5, input_tokens=11, output_tokens=0),
            )

    async def run():
        base = Reverse()
        base.release.clear()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
        task = asyncio.create_task(model.embed(("a", "b", "a")))
        await base.started.wait()
        assert not task.done()
        base.release.set()
        result = await task
        assert result.vectors == ((97.0,), (98.0,), (97.0,))
        assert result.usage == InferenceUsage(requests=10, input_tokens=22, output_tokens=0)

    asyncio.run(run())


@pytest.mark.parametrize("extra", [-1, 1])
def test_invalid_counts_are_not_cached(extra):
    async def run():
        base = Provider()
        base.extra = extra
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        with pytest.raises(ValueError, match="embedding-vector-count"):
            await model.embed(("a", "b"))
        base.extra = 0
        assert len((await model.embed(("a", "b"))).vectors) == 2
        assert base.calls == [("a", "b"), ("a", "b")]

    asyncio.run(run())


def test_cancellation_with_cache_disabled_and_abandoned_failure():
    async def run():
        base = Provider()
        base.release.clear()
        model = CoalescingEmbeddingModel(base, cache_size=0, coalesce_ms=0)
        first = asyncio.create_task(model.embed(("a",)))
        await base.started.wait()
        second = asyncio.create_task(model.embed(("a",)))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        base.release.set()
        result = await second
        assert result.usage.requests == 3
        assert base.calls == [("a",)]
        await model.embed(("a",))
        assert len(base.calls) == 2
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: errors.append(context))
        base.started.clear()
        base.release.clear()
        base.fail = True
        abandoned = asyncio.create_task(model.embed(("bad",)))
        await base.started.wait()
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        base.release.set()
        for _ in range(5):
            await asyncio.sleep(0)
        base.fail = False
        assert (await model.embed(("bad",))).usage.requests == 3
        assert errors == []

    asyncio.run(run())


def test_lru_refresh_raw_identity_and_wrapper_isolation():
    async def run():
        base = Provider()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=2)
        await model.embed(("a", "b"))
        await model.embed(("a",))
        await model.embed(("c",))
        assert (await model.embed(("a",))).usage.requests == 0
        assert (await model.embed(("b",))).usage.requests == 3
        other = CoalescingEmbeddingModel(base, coalesce_ms=0)
        assert (await other.embed(("b",))).usage.requests == 3
        raw = await other.embed(("a", "A", " a", "a "))
        assert raw.vectors == ((97.0,), (65.0,), (129.0,), (129.0,))
        assert base.calls[-1] == ("a", "A", " a", "a ")

    asyncio.run(run())


@pytest.mark.parametrize(
    "options",
    [
        {"max_batch_size": 0},
        {"max_batch_size": 1.5},
        {"cache_size": -1},
        {"coalesce_ms": -1},
        {"coalesce_ms": float("inf")},
        {"coalesce_ms": float("nan")},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(Provider(), **options)


def test_failure_reaches_all_shared_waiters_and_retry_reissues_only_failed_text():
    async def run():
        base = Provider()
        base.fail = True
        base.release.clear()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        first = asyncio.create_task(model.embed(("bad", "bad")))
        await base.started.wait()
        second = asyncio.create_task(model.embed(("bad",)))
        await asyncio.sleep(0)
        base.release.set()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)
        assert base.calls == [("bad",)]
        base.fail = False
        assert (await model.embed(("bad",))).usage.requests == 3
        assert base.calls == [("bad",), ("bad",)]

    asyncio.run(run())
