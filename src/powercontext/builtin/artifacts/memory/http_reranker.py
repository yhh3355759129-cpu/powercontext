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

"""Strict HTTP scoring with one stable global selection across all batches."""

from __future__ import annotations

import math
from typing import cast

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


def _batch_scores(payload: object, first_rank: int, last_rank: int) -> dict[int, int | float]:
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain only a scores array")  # noqa: TRY003
    items = cast(dict[str, object], payload)["scores"]
    if not isinstance(items, list):
        raise ValueError("rerank scores must be an array")  # noqa: TRY003, TRY004
    scores: dict[int, int | float] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each rerank score must contain only rank and score")  # noqa: TRY003
        fields = cast(dict[str, object], item)
        rank, score = fields["rank"], fields["score"]
        if type(rank) is not int or not first_rank <= rank <= last_rank or rank in scores:
            raise ValueError("rerank ranks must cover the current batch exactly once")  # noqa: TRY003
        # bool is an int subclass; never coerce strings or booleans to numbers.
        # Integers are finite without conversion (which could overflow a float).
        if type(score) not in (int, float) or (isinstance(score, float) and not math.isfinite(score)):
            raise ValueError("rerank scores must be finite numbers")  # noqa: TRY003
        scores[rank] = cast(int | float, score)
    if len(scores) != last_rank - first_rank + 1:
        raise ValueError("rerank response is missing batch ranks")  # noqa: TRY003
    return scores


class HTTPMemoryReranker:
    """Use a caller-owned client to score every candidate before global sorting.

    No retries, fallback scores, or client lifecycle management are performed.
    Service and transport errors propagate; malformed responses fail closed.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("rerank batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return original one-based ranks, sorted by score then original rank."""
        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if type(limit) is not int or not 1 <= limit <= len(candidates):
            raise ValueError("rerank limit must be an integer within the candidate count")  # noqa: TRY003
        scores: dict[int, int | float] = {}
        requests = 0
        for offset in range(0, len(candidates), self._batch_size):
            batch = candidates[offset : offset + self._batch_size]
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in enumerate(batch, start=offset + 1)
                    ],
                },
                follow_redirects=False,
            )
            requests += 1
            # Check status before attempting JSON/schema validation. In
            # particular a redirect must not become another scoring request.
            response.raise_for_status()
            scores.update(_batch_scores(response.json(), offset + 1, offset + len(batch)))
        selected = sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit]
        return MemoryRerankDecision(selected_ranks=tuple(selected), usage=InferenceUsage(requests=requests))


__all__ = ["HTTPMemoryReranker"]
