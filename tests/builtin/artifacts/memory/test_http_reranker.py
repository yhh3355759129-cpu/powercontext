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

"""HTTP protocol validation, complete batch coverage and native Memory identity."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker
from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    SearchMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft


def hit(rank: int) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="original", revision=3),
        entry_id=f"entry-{rank}",
        entry_version_id=f"version-{rank}",
        text=f" 原文 {rank} café\n",
        score=1 / rank,
        matched_by=("fts",),
    )


@pytest.mark.parametrize("count,batch_size", [(1, 1), (8, 1), (8, 2), (9, 2), (8, 32), (173, 7)])
@pytest.mark.parametrize("scoring", ["mixed", "ties", "negative"])
def test_scores_all_candidates_once_and_globally_selects(count, batch_size, scoring):
    async def scenario():
        candidates = tuple(hit(rank) for rank in range(1, count + 1))
        snapshot = tuple(candidate.model_dump() for candidate in candidates)
        scores = {
            rank: (rank % 4 if scoring == "mixed" else 0 if scoring == "ties" else -count + rank)
            for rank in range(1, count + 1)
        }
        requests = []
        query = " 原查询 café\n"

        def handler(request):
            assert request.method == "POST" and request.url.path == "/score"
            body = json.loads(request.content)
            assert set(body) == {"query", "candidates"} and body["query"] == query
            assert 1 <= len(body["candidates"]) <= batch_size
            for item in body["candidates"]:
                assert set(item) == {"rank", "text"}
                assert type(item["rank"]) is int
                assert item["text"] == candidates[item["rank"] - 1].text
            requests.append(body)
            return httpx.Response(
                200,
                json={
                    "scores": [
                        {"rank": item["rank"], "score": scores[item["rank"]]} for item in reversed(body["candidates"])
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker("https://reranker.invalid/score", client=client, batch_size=batch_size)
            for _ in range(2):
                start = len(requests)
                limit = min(count, 10)
                decision = await reranker.rerank(query, candidates, limit)
                assert decision.selected_ranks == tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit])
                assert decision.usage.requests == len(requests) - start == math.ceil(count / batch_size)
                assert [item["rank"] for body in requests[start:] for item in body["candidates"]] == list(
                    range(1, count + 1)
                )
                assert not decision.used_fallback and decision.discarded_rank_count == 0
                assert not client.is_closed
            assert reranker.policy_id == "http-cross-encoder-v1"
        assert tuple(candidate.model_dump() for candidate in candidates) == snapshot

    asyncio.run(scenario())


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.0, "2", None])
def test_bad_batch_sizes_fail_before_requests(batch_size):
    async def scenario():
        def forbidden(request):
            raise AssertionError("invalid parameters sent a request")  # noqa: TRY003

        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            with pytest.raises(ValueError):
                HTTPMemoryReranker("https://invalid", client=client, batch_size=batch_size)

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", [0, -1, 3, True, False, 1.0, "1", None])
def test_bad_limits_fail_before_requests(limit):
    async def scenario():
        def forbidden(request):
            raise AssertionError("invalid parameters sent a request")  # noqa: TRY003

        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (hit(1), hit(2)), limit)

    asyncio.run(scenario())


def test_empty_candidates_fail_before_requests():
    async def scenario():
        def forbidden(request):
            raise AssertionError("empty candidate pool sent a request")  # noqa: TRY003

        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "scores",
        42,
        True,
        {},
        {"scores": []},
        {"scores": None},
        {"scores": {}},
        {"scores": [{"rank": 1, "score": 1}], "extra": True},
        {"scores": [None]},
        {"scores": [[]]},
        {"scores": [1]},
        {"scores": [{}]},
        {"scores": [{"rank": 1}]},
        {"scores": [{"score": 1}]},
        {"scores": [{"rank": 1, "score": 1, "extra": 0}]},
        {"scores": [{"rank": 0, "score": 1}]},
        {"scores": [{"rank": 2, "score": 1}]},
        {"scores": [{"rank": 1, "score": 1}, {"rank": 1, "score": 2}]},
        *[{"scores": [{"rank": rank, "score": 1}]} for rank in (True, False, 1.0, "1", None, [], {})],
        *[
            {"scores": [{"rank": 1, "score": score}]}
            for score in (True, False, "1", None, [], {}, float("nan"), float("inf"), -float("inf"))
        ],
    ],
)
def test_invalid_schema_is_not_coerced(payload):
    async def scenario():
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, content=json.dumps(payload).encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (hit(1),), 1)
            assert len(calls) == 1 and not client.is_closed

    asyncio.run(scenario())


@pytest.mark.parametrize("content", [b"", b"{bad", b"[]junk", b"\xff", b'{"scores":[{"rank":1,"score":1e999}]}'])
def test_invalid_json_and_overflow_scores_fail(content):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
        ) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (hit(1),), 1)

    asyncio.run(scenario())


def test_cross_batch_rank_is_rejected_without_retry_or_partial_decision():
    async def scenario():
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            rank = calls[-1]["candidates"][0]["rank"]
            return httpx.Response(200, json={"scores": [{"rank": 1 if rank == 2 else rank, "score": rank}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://invalid", client=client, batch_size=1).rerank(
                    "q", tuple(hit(i) for i in range(1, 4)), 1
                )
        assert len(calls) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [201, 204, 301, 302, 307, 308, 400, 429, 500, 503])
def test_status_is_checked_before_schema_and_redirects_are_never_followed(status):
    async def scenario():
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(
                status, content=b"invalid json", headers={"Location": "https://redirect.invalid/score"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            # All 2xx statuses enter schema validation; all others retain HTTPStatusError.
            exception = ValueError if 200 <= status < 300 else httpx.HTTPStatusError
            with pytest.raises(exception) as caught:
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (hit(1),), 1)
            if isinstance(caught.value, httpx.HTTPStatusError):
                assert caught.value.response.status_code == status
            assert len(calls) == 1 and not client.is_closed

    asyncio.run(scenario())


@pytest.mark.parametrize("error_class", [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_transport_errors_are_preserved_and_not_retried(error_class):
    async def scenario():
        errors = []

        def handler(request):
            error = error_class("injected transport failure", request=request)
            errors.append(error)
            raise error

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(error_class) as caught:
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (hit(1),), 1)
            assert len(errors) == 1 and caught.value is errors[0] and not client.is_closed

    asyncio.run(scenario())


def test_large_finite_integer_scores_are_not_rounded_or_overflowed():
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    content=json.dumps({
                        "scores": [{"rank": 1, "score": 10**400}, {"rank": 2, "score": 10**400 + 1}]
                    }).encode(),
                )
            )
        ) as client:
            decision = await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (hit(1), hit(2)), 2)
            assert decision.selected_ranks == (2, 1)

    asyncio.run(scenario())


def test_real_runtime_search_preserves_every_entry_version_citation_and_usage(tmp_path: Path):
    async def scenario():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'rerank.db'}"))
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Rerank", summary="Offline HTTP scoring", idempotency_key="rerank")
            )
            memory = runtime.memory.for_scope(scope.scope_id)
            remembered = await memory.remember(
                RememberMemoryRequest(
                    entries=tuple(
                        MemoryEntryInput(kind="fact", text=f"Project evidence candidate {rank} café.")
                        for rank in range(17)
                    )
                )
            )
            baseline = await memory.search(SearchMemoryRequest(query="project", limit=30, mode="fts"))
            assert len(baseline.hits) == 17
        calls = []

        def handler(request):
            body = json.loads(request.content)
            assert body["query"] == "project"
            calls.append(body)
            return httpx.Response(
                200,
                json={
                    "scores": [
                        {"rank": item["rank"], "score": item["rank"] // 2} for item in reversed(body["candidates"])
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker("https://invalid", client=client, batch_size=7, policy_id="test-policy")
            async with open_builtin_runtime(config, memory_reranker=reranker) as runtime:
                memory = runtime.memory.for_scope(scope.scope_id)
                for _ in range(2):
                    start = len(calls)
                    result = await memory.search(SearchMemoryRequest(query="project", limit=4, mode="fts"))
                    assert result.memory_ref == remembered.memory_ref
                    trace = result.rerank
                    assert trace is not None and trace.policy_id == "test-policy"
                    assert trace.candidate_hits == baseline.hits
                    expected = tuple(sorted(range(1, 18), key=lambda rank: (-(rank // 2), rank))[:4])
                    assert trace.selected_ranks == expected
                    assert result.hits == tuple(baseline.hits[rank - 1] for rank in expected)
                    assert trace.usage.requests == len(calls) - start == 3
                    assert [item for body in calls[start:] for item in body["candidates"]] == [
                        {"rank": rank, "text": candidate.text} for rank, candidate in enumerate(baseline.hits, 1)
                    ]
                # Read every original exact citation, not just the selected hits.
                for candidate in baseline.hits:
                    citation = MemoryCitation(
                        memory_ref=candidate.memory_ref,
                        entry_id=candidate.entry_id,
                        entry_version_id=candidate.entry_version_id,
                    )
                    entry = await memory.get(GetMemoryEntryRequest(citation=citation))
                    assert entry.memory_ref == candidate.memory_ref
                    assert entry.entry.entry_id == candidate.entry_id
                    assert entry.entry.entry_version_id == candidate.entry_version_id
                    assert entry.entry.text == candidate.text
            assert not client.is_closed
        async with open_builtin_runtime(config) as runtime:
            restored = await runtime.memory.for_scope(scope.scope_id).search(
                SearchMemoryRequest(query="project", limit=30, mode="fts")
            )
            assert restored == baseline

    asyncio.run(scenario())
