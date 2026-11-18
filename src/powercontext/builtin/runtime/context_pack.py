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

"""Canonical UTF-8 context packs and independently verifiable prefix deltas."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ContextPackError(ValueError):
    """Invalid context-pack data, chunking bounds, or reconstruction."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _validate_items(items: tuple[bytes, ...]) -> None:
    if type(items) is not tuple or any(type(item) is not bytes for item in items):
        raise ContextPackError("chunks must be a tuple of bytes")  # noqa: TRY003
    try:
        for item in items:
            item.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContextPackError("each chunk must be valid UTF-8") from error  # noqa: TRY003


def _validate_digest(value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContextPackError("digest must be sha256:<64 lowercase hex digits>")  # noqa: TRY003


def _encoded(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decoded(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("wire chunks must be a Base64 string array")  # noqa: TRY003
    result = []
    for encoded in value:
        if not isinstance(encoded, str):
            raise ContextPackError("wire chunks must be Base64 strings")  # noqa: TRY003
        try:
            item = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ContextPackError("invalid Base64 chunk") from error  # noqa: TRY003
        if base64.b64encode(item).decode("ascii") != encoded:
            raise ContextPackError("chunks must use canonical standard Base64")  # noqa: TRY003
        result.append(item)
    items = tuple(result)
    _validate_items(items)
    return items


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContextPackError("duplicate JSON field")  # noqa: TRY003
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContextPackError("non-finite JSON values are not valid protocol fields")  # noqa: TRY003


def _read_wire(raw: bytes, schema: str, keys: set[str]) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ContextPackError("wire input must be bytes")  # noqa: TRY003
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        raise ContextPackError("invalid UTF-8 JSON wire data") from error  # noqa: TRY003
    if not isinstance(value, dict) or set(value) != keys or value["schema"] != schema:
        raise ContextPackError("wrong schema or wire fields")  # noqa: TRY003
    return value


@dataclass(frozen=True, slots=True)
class ContextPack:
    """Immutable complete chunks with canonical digest and exact wire accounting."""

    items: tuple[bytes, ...]
    digest: str = field(init=False)
    _wire: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_items(self.items)
        unsigned = {"schema": _PACK_SCHEMA, "items": _encoded(self.items)}
        digest = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
        # The byte_count field contributes its own decimal digits to the size.
        count = 0
        while True:
            wire = _canonical({**unsigned, "digest": digest, "byte_count": count})
            if len(wire) == count:
                break
            count = len(wire)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "_wire", wire)

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        return cls(items)

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if type(chunk_bytes) is not int or chunk_bytes < 1 or not isinstance(content, str):
            raise ContextPackError("content must be text and chunk_bytes a positive integer")  # noqa: TRY003
        try:
            raw = content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ContextPackError("content is not valid UTF-8 text") from error  # noqa: TRY003
        items = []
        offset = 0
        while offset < len(raw):
            end = min(offset + chunk_bytes, len(raw))
            while end < len(raw) and raw[end] & 0xC0 == 0x80:
                end -= 1
            if end == offset:
                raise ContextPackError("chunk_bytes cannot contain a UTF-8 character")  # noqa: TRY003
            items.append(raw[offset:end])
            offset = end
        return cls.build(tuple(items))

    @property
    def byte_count(self) -> int:
        return len(self._wire)

    def to_bytes(self) -> bytes:
        return self._wire

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPack:
        value = _read_wire(raw, _PACK_SCHEMA, {"schema", "items", "digest", "byte_count"})
        pack = cls.build(_decoded(value["items"]))
        if value["digest"] != pack.digest:
            raise ContextPackError("pack digest does not match chunks")  # noqa: TRY003
        if type(value["byte_count"]) is not int or value["byte_count"] != pack.byte_count:
            raise ContextPackError("pack byte_count does not match canonical wire bytes")  # noqa: TRY003
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """One common base prefix followed by the target's unchanged suffix chunks."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.base_digest)
        _validate_digest(self.target_digest)
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        _validate_items(self.appended)

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta requires complete base and target packs")  # noqa: TRY003
        prefix = 0
        for old, new in zip(base.items, target.items, strict=False):
            if old != new:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    def to_bytes(self) -> bytes:
        return _canonical({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded(self.appended),
        })

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPackDelta:
        value = _read_wire(raw, _DELTA_SCHEMA, {"schema", "base_digest", "target_digest", "prefix_items", "appended"})
        return cls(value["base_digest"], value["target_digest"], value["prefix_items"], _decoded(value["appended"]))


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Reconstruct without sender state and verify both ends of the transfer."""
    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("apply requires a complete base pack and delta")  # noqa: TRY003
    if base.digest != delta.base_digest:
        raise ContextPackError("delta base digest does not match")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base chunk count")  # noqa: TRY003
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match reconstructed chunks")  # noqa: TRY003
    return target


__all__ = ["ContextPack", "ContextPackDelta", "ContextPackError", "apply_context_pack_delta"]
