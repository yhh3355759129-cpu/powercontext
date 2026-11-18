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

"""Event-loop-local embedding request sharing and bounded vector caching."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(eq=False)
class _BatchUsage:
    usage: InferenceUsage


@dataclass(frozen=True)
class _EmbeddedText:
    vector: EmbeddingVector
    batch: _BatchUsage


def _consume_exception(future: asyncio.Future[_EmbeddedText]) -> None:
    # A caller may leave before its provider finishes, including on cancellation.
    if not future.cancelled():
        future.exception()


class CoalescingEmbeddingModel:
    """Share pending texts on one event loop and cache successful immutable vectors.

    Usage belongs to the provider batch. Each caller counts every batch it waits
    for once, even if it contributes several texts or repeated input positions.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size")
        if not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms")
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[_EmbeddedText]] = {}
        self._pending: dict[str, asyncio.Future[_EmbeddedText]] = {}
        self._flush_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def profile(self) -> EmbeddingProfile:
        """Expose the deployment profile of the wrapped model."""
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Embed texts in original order, preserving repeats and provider usage."""
        cached: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[_EmbeddedText]] = {}
        loop = asyncio.get_running_loop()
        for text in texts:
            if text in self._cache:
                cached[text] = self._cache[text]
                self._cache.move_to_end(text)
            else:
                future = self._inflight.get(text)
                if future is None:
                    future = loop.create_future()
                    future.add_done_callback(_consume_exception)
                    self._inflight[text] = future
                    self._pending[text] = future
                waiting[text] = future
        if self._pending and self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush())
        # Only the per-caller shields are cancelled when this caller leaves.
        results = await asyncio.gather(*(asyncio.shield(future) for future in waiting.values()))
        batches = {result.batch for result in results}
        vectors = dict(cached)
        vectors.update((text, result.vector) for text, result in zip(waiting, results, strict=True))
        usage = InferenceUsage(requests=sum(batch.usage.requests for batch in batches))
        for field in ("input_tokens", "output_tokens"):
            values = [getattr(batch.usage, field) for batch in batches]
            known = [value for value in values if value is not None]
            setattr(usage, field, sum(known) if known else None)
        return EmbeddingResult(vectors=tuple(vectors[text] for text in texts), usage=usage)

    async def _flush(self) -> None:
        # Even a zero window yields so concurrent callers can register their texts.
        await asyncio.sleep(self._coalesce_seconds)
        pending = self._pending
        self._pending = {}
        self._flush_task = None
        items = list(pending.items())
        for start in range(0, len(items), self._max_batch_size):
            task = asyncio.create_task(self._run_batch(dict(items[start : start + self._max_batch_size])))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_batch(self, pending: dict[str, asyncio.Future[_EmbeddedText]]) -> None:
        try:
            result = await self._model.embed(tuple(pending))
            if len(result.vectors) != len(pending):
                raise ValueError("embedding-vector-count")  # noqa: TRY301 - batch waiters receive this error
            batch = _BatchUsage(result.usage.model_copy())
            for (text, future), vector in zip(pending.items(), result.vectors, strict=True):
                immutable = tuple(vector)
                if self._cache_size:
                    self._cache[text] = immutable
                    self._cache.move_to_end(text)
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                future.set_result(_EmbeddedText(immutable, batch))
        except asyncio.CancelledError:
            for future in pending.values():
                future.cancel()
            raise
        except Exception as error:
            for future in pending.values():
                future.set_exception(error)
        finally:
            for text in pending:
                del self._inflight[text]
