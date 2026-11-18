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

"""Supplementary Runtime wire-cost measurement; not the unpublished official schedule.

Run with the competition Python and PYTHONPATH=src. Accepts an original QMSum
meeting JSON or a turns.json-shaped file. Input bytes and all settings are
reported so the supplementary measurement can be reproduced and audited.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, CaptureSource, PrepareContextRequest, open_builtin_runtime
from powercontext.builtin.runtime.context_pack import ContextPack, ContextPackDelta, apply_context_pack_delta
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource


class TurnPipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


def select_turns(path: Path, terms: list[str]) -> tuple[str, list[tuple[int, str]]]:
    raw = path.read_bytes()
    data = json.loads(raw)
    turns = []
    for index, turn in enumerate(data.get("meeting_transcripts", data.get("turns", []))):
        text = (turn["speaker"] + ": " + turn["content"]) if "speaker" in turn else turn["text"]
        if 120 <= len(text.encode("utf-8")) <= 900 and any(term.lower() in text.lower() for term in terms):
            turns.append((turn.get("turn_index", index), text))
    return hashlib.sha256(raw).hexdigest(), turns


def measurement_turns(args: argparse.Namespace) -> tuple[str, list[tuple[int, str]]]:
    digest, turns = select_turns(args.input, args.terms)
    if args.expected_sha256 is not None and digest != args.expected_sha256:
        raise ValueError("input checksum mismatch")  # noqa: TRY003
    needed = args.initial_turns + len(args.append_rounds)
    if len(turns) < needed:
        raise ValueError("not enough original turns; no repeated or synthesized substitutes")  # noqa: TRY003
    if (
        args.rounds < 1
        or args.initial_turns < 1
        or sorted(set(args.append_rounds)) != args.append_rounds
        or any(not 0 < n < args.rounds for n in args.append_rounds)
    ):
        raise ValueError("invalid measurement schedule")  # noqa: TRY003
    return digest, turns


async def measure(args: argparse.Namespace) -> dict[str, Any]:
    digest, turns = measurement_turns(args)
    needed = args.initial_turns + len(args.append_rounds)
    rows: list[dict[str, Any]] = []
    contents: set[bytes] = set()
    with TemporaryDirectory(prefix="pc-pack-cost-") as directory:
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{Path(directory) / 'runtime.db'}"))
        async with open_builtin_runtime(config, candidate_pipeline=TurnPipeline()) as runtime:
            if runtime.scopes is None:
                raise RuntimeError("Runtime has no Scope service")  # noqa: TRY003
            scope = await runtime.scopes.create(
                ScopeDraft(
                    title="Context wire measurement",
                    summary="Supplementary QMSum schedule",
                    idempotency_key="context-cost",
                )
            )
            sources = runtime.sources.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            context = runtime.context.for_scope(scope.scope_id)
            imported = 0
            receiver = None
            for round_index in range(args.rounds):
                wanted = args.initial_turns + sum(n <= round_index for n in args.append_rounds)
                for index, text in turns[imported:wanted]:
                    await sources.capture(
                        CaptureSource(source_id=f"meeting-turn-{index}", content=text, metadata={"turn_index": index})
                    )
                imported = wanted
                while (await memory.flush()).processed:
                    pass
                query = args.switch_query if round_index >= args.switch_round else args.query
                request = PrepareContextRequest(query=query, max_bytes=args.max_bytes)
                ordinary = await context.prepare(request)
                if not ordinary.content:
                    raise RuntimeError(f"empty ordinary prepare at round {round_index}")  # noqa: TRY003
                body = ordinary.content.encode("utf-8")
                contents.add(body)
                sent = await context.prepare_incremental(request, base_pack=receiver)
                wire = sent.to_bytes()
                if isinstance(sent, ContextPack):
                    receiver = ContextPack.from_bytes(wire)
                else:
                    if receiver is None:
                        raise RuntimeError("delta without receiver base")  # noqa: TRY003
                    receiver = apply_context_pack_delta(
                        ContextPack.from_bytes(receiver.to_bytes()), ContextPackDelta.from_bytes(wire)
                    )
                if b"".join(receiver.items) != body:
                    raise RuntimeError(f"receiver did not reproduce ordinary prepare at round {round_index}")  # noqa: TRY003
                rows.append({
                    "round": round_index,
                    "query": query,
                    "imported_turns": wanted,
                    "raw_bytes": len(body),
                    "full_pack_bytes": ContextPack.from_content(ordinary.content).byte_count,
                    "gzip_bytes": len(gzip.compress(body, compresslevel=9, mtime=0)),
                    "incremental_bytes": len(wire),
                    "kind": "pack" if isinstance(sent, ContextPack) else "delta",
                })
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("raw_bytes", "full_pack_bytes", "gzip_bytes", "incremental_bytes")
    }
    sent_bytes = totals["incremental_bytes"]
    return {
        "measurement": "supplementary; official 40-round schedule not supplied",
        "input_sha256": digest,
        "configuration": {
            "rounds": args.rounds,
            "initial_turns": args.initial_turns,
            "append_rounds_zero_based": args.append_rounds,
            "query": args.query,
            "switch_query": args.switch_query,
            "switch_round_zero_based": args.switch_round,
            "selection_terms": args.terms,
            "turn_utf8_bytes_inclusive": [120, 900],
            "max_bytes": args.max_bytes,
            "chunk_bytes": 512,
            "gzip": {"compresslevel": 9, "mtime": 0, "shared_state": False},
            "turn_indices": [i for i, _ in turns[:needed]],
        },
        "nonempty_rounds": len(rows),
        "distinct_contexts": len(contents),
        "totals": totals,
        "savings_vs_raw": 1 - sent_bytes / totals["raw_bytes"],
        "savings_vs_full_pack": 1 - sent_bytes / totals["full_pack_bytes"],
        "savings_vs_gzip": 1 - sent_bytes / totals["gzip_bytes"],
        "rounds": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--terms", nargs="+", default=["remote", "button", "design", "control"])
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--initial-turns", type=int, default=8)
    parser.add_argument("--append-rounds", type=int, nargs="*", default=[10, 20, 30])
    parser.add_argument("--query", default="remote")
    parser.add_argument("--switch-query", default="control")
    parser.add_argument("--switch-round", type=int, default=32)
    parser.add_argument("--max-bytes", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(measure(args))
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
