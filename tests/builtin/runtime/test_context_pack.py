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

"""Protocol boundaries and independent reconstruction of ordinary Runtime output."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    PrepareContextRequest,
    RememberMemoryRequest,
    SearchMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.context_pack import (
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)
from powercontext.builtin.scope import ScopeDraft


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 511, 512, 513])
def test_deterministic_utf8_chunks(size):
    text = ('a\r\n\t\\"' + ("é" if size >= 2 else "") + ("海" if size >= 3 else "") + ("🌊" if size >= 4 else "")) * 200
    pack = ContextPack.from_content(text, chunk_bytes=size)
    assert pack == ContextPack.from_content(text, chunk_bytes=size)
    assert all(0 < len(item) <= size and item.decode().encode() == item for item in pack.items)
    assert b"".join(pack.items) == text.encode()
    assert ContextPack.from_bytes(pack.to_bytes()) == pack
    assert pack.byte_count == len(pack.to_bytes())


@pytest.mark.parametrize("size", [True, False, 0, -1, 1.5, "512", None])
def test_invalid_chunk_sizes_fail_even_for_empty_content(size):
    with pytest.raises(ContextPackError):
        ContextPack.from_content("", chunk_bytes=size)


@pytest.mark.parametrize("text,size", [("é", 1), ("海", 2), ("🌊", 3), ("a海", 2), ("\ud800", 512)])
def test_unrepresentable_utf8_chunk_is_rejected(text, size):
    with pytest.raises(ContextPackError):
        ContextPack.from_content(text, chunk_bytes=size)


@pytest.mark.parametrize("items", [[b"a"], (bytearray(b"a"),), ("text",), (b"\xff",), (b"\xe6\xb5", b"\xb7"), None])
def test_build_rejects_non_bytes_or_invalid_individual_chunks(items):
    with pytest.raises(ContextPackError):
        ContextPack.build(items)


def test_exact_fixed_point_size_and_independent_digest_across_decimal_widths():
    widths = set()
    for length in range(0, 9000, 7):
        pack = ContextPack.from_content("x" * length)
        payload = json.loads(pack.to_bytes())
        assert payload["byte_count"] == len(pack.to_bytes()) == pack.byte_count
        widths.add(len(str(pack.byte_count)))
        unsigned = {"schema": payload["schema"], "items": payload["items"]}
        assert pack.digest == "sha256:" + hashlib.sha256(wire(unsigned)).hexdigest()
        assert pack.to_bytes() == wire(payload)
    assert widths >= {3, 4, 5}


@pytest.mark.parametrize(
    "target",
    [(), (b"a",), (b"a", b"b"), (b"a", b"b", b"c"), (b"new", b"b", b"c"), (b"a", b"new", b"c"), (b"a", b"b", b"tail")],
)
def test_all_positions_shortening_and_empty_transitions(target):
    for base_items in ((), (b"a", b"b", b"c")):
        base = ContextPack.build(base_items)
        wanted = ContextPack.build(target)
        delta = ContextPackDelta.between(base, wanted)
        received = ContextPack.from_bytes(base.to_bytes())
        rebuilt = apply_context_pack_delta(received, ContextPackDelta.from_bytes(delta.to_bytes()))
        assert rebuilt == wanted and rebuilt.to_bytes() == wanted.to_bytes()
        assert delta.byte_count == len(delta.to_bytes())
        assert "byte_count" not in json.loads(delta.to_bytes())
        # A shorter valid shared prefix (including zero) also interoperates.
        for prefix in range(delta.prefix_items + 1):
            alternate = ContextPackDelta(base.digest, wanted.digest, prefix, target[prefix:])
            assert apply_context_pack_delta(received, ContextPackDelta.from_bytes(alternate.to_bytes())) == wanted


@pytest.mark.parametrize("kind", ["pack", "delta"])
def test_missing_extra_wrong_schema_and_wire_types(kind):
    pack = ContextPack.build((b"hello",))
    delta = ContextPackDelta.between(pack, ContextPack.build((b"hello", b"world")))
    obj = pack if kind == "pack" else delta
    payload = json.loads(obj.to_bytes())
    for key in payload:
        with pytest.raises(ContextPackError):
            obj.from_bytes(wire({k: v for k, v in payload.items() if k != key}))
    for update in ({"extra": 1}, {"schema": "other"}, {"schema": None}):
        with pytest.raises(ContextPackError):
            obj.from_bytes(wire(payload | update))
    for raw in (
        b"",
        b"\xff",
        b"{bad",
        b"null",
        b"[]",
        b"true",
        b"42",
        b'{"byte_count":' + b"1" * 5000 + b"}",
        obj.to_bytes().decode(),
        bytearray(obj.to_bytes()),
    ):
        with pytest.raises(ContextPackError):
            obj.from_bytes(cast(bytes, raw))
    repeated = obj.to_bytes().replace(b"{", b'{"schema":"duplicate",', 1)
    with pytest.raises(ContextPackError):
        obj.from_bytes(repeated)


@pytest.mark.parametrize(
    "chunks",
    [
        None,
        {},
        "YQ==",
        [True],
        [1],
        [None],
        [[]],
        ["!"],
        ["YQ"],
        ["YQ=== "],
        ["YQ==\n"],
        ["_w=="],
        ["YR=="],
        ["/w=="],
        ["5rU=", "tw=="],
    ],
)
def test_bad_base64_or_utf8_rejected_in_both_decoders(chunks):
    base = ContextPack.build((b"hello",))
    delta = ContextPackDelta.between(base, ContextPack.build((b"world",)))
    for obj, key in ((base, "items"), (delta, "appended")):
        payload = json.loads(obj.to_bytes())
        with pytest.raises(ContextPackError):
            obj.from_bytes(wire(payload | {key: chunks}))


@pytest.mark.parametrize("bad", [None, True, 1, [], "sha256:" + "A" * 64, "sha256:" + "z" * 64, "sha256:123", ""])
def test_wrong_digest_types_and_formats(bad):
    base = ContextPack.build((b"a",))
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(wire(json.loads(base.to_bytes()) | {"digest": bad}))
    delta = ContextPackDelta.between(base, base)
    for key in ("base_digest", "target_digest"):
        with pytest.raises(ContextPackError):
            ContextPackDelta.from_bytes(wire(json.loads(delta.to_bytes()) | {key: bad}))


@pytest.mark.parametrize("count", [True, False, 0, -1, 1.5, "160", None, 999999])
def test_invalid_byte_count_is_never_coerced(count):
    pack = ContextPack.build(())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(wire(json.loads(pack.to_bytes()) | {"byte_count": count}))


@pytest.mark.parametrize("prefix", [True, False, -1, 1.5, "1", None])
def test_prefix_types_are_strict_in_constructor_and_decoder(prefix):
    base = ContextPack.build((b"a",))
    with pytest.raises(ContextPackError):
        ContextPackDelta(base.digest, base.digest, prefix, ())
    delta = ContextPackDelta.between(base, base)
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(wire(json.loads(delta.to_bytes()) | {"prefix_items": prefix}))


def test_wrong_base_target_and_out_of_bounds_prefix():
    base = ContextPack.build((b"a",))
    target = ContextPack.build((b"a", b"b"))
    delta = ContextPackDelta.between(base, target)
    for other, change in (
        (target, delta),
        (base, ContextPackDelta(base.digest, target.digest, 2, ())),
        (base, ContextPackDelta(base.digest, base.digest, 1, (b"b",))),
    ):
        with pytest.raises(ContextPackError):
            apply_context_pack_delta(other, change)


def test_separate_optimized_python_receiver_needs_only_serialized_bytes():
    base = ContextPack.from_content("海洋\n" * 100)
    target = ContextPack.from_content("海洋\n" * 100 + "new tail")
    delta = ContextPackDelta.between(base, target)
    program = """
import base64, json, sys
from powercontext.builtin.runtime.context_pack import ContextPack, ContextPackDelta, apply_context_pack_delta
base, delta = [base64.b64decode(s) for s in json.load(sys.stdin)]
pack = apply_context_pack_delta(ContextPack.from_bytes(base), ContextPackDelta.from_bytes(delta))
sys.stdout.buffer.write(b''.join(pack.items))
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        input=wire([base64.b64encode(base.to_bytes()).decode(), base64.b64encode(delta.to_bytes()).decode()]),
        capture_output=True,
        check=True,
    )
    assert result.stdout == b"".join(target.items)


def test_runtime_append_query_switch_empty_and_citations_survive_transport(tmp_path: Path):
    async def scenario():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'packs.db'}"))
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(ScopeDraft(title="Packs", summary="Transport", idempotency_key="packs"))
            memory = runtime.memory.for_scope(scope.scope_id)
            context = runtime.context.for_scope(scope.scope_id)
            request = PrepareContextRequest(query="ocean", max_bytes=4096)
            assert (await context.prepare(request)).content is None
            first = await context.prepare_incremental(request)
            assert isinstance(first, ContextPack) and first.items == ()
            receiver = ContextPack.from_bytes(first.to_bytes())
            await memory.remember(
                RememberMemoryRequest(
                    entries=(
                        MemoryEntryInput(kind="fact", text="Ocean evidence 海洋."),
                        MemoryEntryInput(kind="fact", text="Mountain evidence 山峰."),
                    )
                )
            )
            for query in ("ocean", "mountain", "unmatched", "ocean"):
                current = PrepareContextRequest(query=query, max_bytes=4096)
                ordinary = await context.prepare(current)
                changed = await context.prepare_incremental(current, base_pack=receiver)
                assert isinstance(changed, ContextPackDelta)
                receiver = apply_context_pack_delta(
                    ContextPack.from_bytes(receiver.to_bytes()), ContextPackDelta.from_bytes(changed.to_bytes())
                )
                assert b"".join(receiver.items) == (ordinary.content or "").encode()
                page = await memory.search(SearchMemoryRequest(query=query, mode="fts"))
                for hit in page.hits:
                    cited = await memory.get(
                        GetMemoryEntryRequest(
                            citation=MemoryCitation(
                                memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id
                            )
                        )
                    )
                    assert cited.entry.text == hit.text
                    assert (
                        ordinary.content
                        and hit.entry_id in ordinary.content
                        and hit.entry_version_id in ordinary.content
                    )
            await memory.remember(
                RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text="Ocean appendix evidence."),))
            )
            ordinary = await context.prepare(request)
            update = await context.prepare_incremental(request, base_pack=receiver)
            assert isinstance(update, ContextPackDelta)
            restored = apply_context_pack_delta(receiver, ContextPackDelta.from_bytes(update.to_bytes()))
            assert ordinary.content is not None
            assert b"".join(restored.items) == ordinary.content.encode()
            calls = []

            async def authorize(scopes):
                calls.append(scopes)
                raise PermissionError

            with pytest.raises(PermissionError):
                await context.prepare_incremental(request, authorize_scopes=authorize)
            assert calls == [(scope.scope_id,)]

    asyncio.run(scenario())
