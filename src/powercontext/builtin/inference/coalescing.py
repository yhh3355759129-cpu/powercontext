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

"""Event-loop-local embedding coalescing, in-flight sharing and vector LRU."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(frozen=True, slots=True)
class _VectorResult:
    vector: EmbeddingVector
    batch: object
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Share exact text computations without sharing a caller's cancellation.

    Registration and dispatch bookkeeping contain no awaits, so callers in one
    event loop need no lock. Independent provider batches execute concurrently.
    Completed-cache entries contain only immutable vectors, never historical
    usage; each waiter receives usage for the batches it actually awaited.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, int | float):
            raise ValueError("coalesce_ms must be a finite non-negative number")  # noqa: TRY003, TRY004
        try:
            delay = float(coalesce_ms)
        except OverflowError as error:
            raise ValueError("coalesce_ms must be finite") from error  # noqa: TRY003
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("coalesce_ms must be a finite non-negative number")  # noqa: TRY003
        self.profile = model.profile
        self._model = model
        self._max_batch_size = max_batch_size
        self._delay = delay / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[_VectorResult]] = {}
        self._pending: dict[str, asyncio.Future[_VectorResult]] = {}
        self._dispatch_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return original order and sum each waited provider batch only once."""
        vectors: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[_VectorResult]] = {}
        loop = asyncio.get_running_loop()
        for text in texts:
            if text in vectors:
                self._cache.move_to_end(text)
                continue
            if text in waiting:
                continue
            if text in self._cache:
                vectors[text] = self._cache[text]
                self._cache.move_to_end(text)
                continue
            future = self._inflight.get(text)
            if future is None:
                future = loop.create_future()
                self._inflight[text] = self._pending[text] = future
            waiting[text] = future
        if self._pending and self._dispatch_task is None:
            self._dispatch_task = asyncio.create_task(self._dispatch())
        # Shield each shared Future, not the provider under a global lock.
        completed = await asyncio.gather(*(asyncio.shield(future) for future in waiting.values()))
        batches: dict[object, InferenceUsage] = {}
        for text, result in zip(waiting, completed, strict=True):
            vectors[text] = result.vector
            batches[result.batch] = result.usage
        usages = tuple(batches.values())
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in texts),
            usage=InferenceUsage(
                requests=sum(usage.requests for usage in usages),
                input_tokens=_sum_tokens(tuple(usage.input_tokens for usage in usages)),
                output_tokens=_sum_tokens(tuple(usage.output_tokens for usage in usages)),
            ),
        )

    async def _dispatch(self) -> None:
        try:
            # Even a zero window yields once, so concurrently registered calls
            # can share work when the provider itself completes synchronously.
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError as error:
            entries = tuple(self._pending.items())
            self._pending.clear()
            self._dispatch_task = None
            self._fail(entries, error)
            return
        entries = tuple(self._pending.items())
        self._pending.clear()
        self._dispatch_task = None
        for start in range(0, len(entries), self._max_batch_size):
            task = asyncio.create_task(self._run_batch(entries[start : start + self._max_batch_size]))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_batch(self, entries: tuple[tuple[str, asyncio.Future[_VectorResult]], ...]) -> None:
        try:
            result = await self._model.embed(tuple(text for text, _ in entries))
            if len(result.vectors) != len(entries):
                raise ValueError("provider vector count must match its text batch")  # noqa: TRY003, TRY301
            # Snapshot provider-owned containers before populating any cache.
            vectors = tuple(tuple(vector) for vector in result.vectors)
            usage = result.usage.model_copy(deep=True)
        except (Exception, asyncio.CancelledError) as error:
            self._fail(entries, error)
            return
        batch = object()
        for (text, future), vector in zip(entries, vectors, strict=True):
            self._inflight.pop(text)
            if self._cache_size:
                self._cache[text] = vector
                self._cache.move_to_end(text)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            future.set_result(_VectorResult(vector, batch, usage))

    def _fail(self, entries: tuple[tuple[str, asyncio.Future[_VectorResult]], ...], error: BaseException) -> None:
        for text, future in entries:
            self._inflight.pop(text, None)
            future.set_exception(error)
            # A canceled caller may no longer await this Future. Mark its
            # exception retrieved immediately; active waiters still receive it.
            future.exception()


def _sum_tokens(values: tuple[int | None, ...]) -> int | None:
    known = tuple(value for value in values if value is not None)
    return sum(known) if known else None


__all__ = ["CoalescingEmbeddingModel"]
