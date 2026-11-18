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

"""Durable per-scope evidence graph rows and Topic eligibility predicates."""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE as NODES


def node_key(node_id: str) -> str:
    return hashlib.sha256(node_id.encode("utf-8")).hexdigest()


async def load_nodes(connection: AsyncConnection, scope_id: str) -> tuple[dict[str, Any], ...]:
    rows = (
        await connection.execute(select(NODES).where(NODES.c.scope_id == scope_id).order_by(NODES.c.ordinal))
    ).mappings()
    return tuple(dict(row) for row in rows)


async def save_node(
    connection: AsyncConnection,
    scope_id: str,
    node_id: str,
    *,
    payload: bytes,
    available: bool,
    kind: str,
    artifact_id: str | None,
    ordinal: int | None = None,
) -> None:
    values = {"payload": payload, "available": available, "kind": kind, "artifact_id": artifact_id}
    if ordinal is not None:
        await connection.execute(
            insert(NODES).values(scope_id=scope_id, node_key=node_key(node_id), ordinal=ordinal, **values)
        )
    else:
        await connection.execute(
            update(NODES).where(NODES.c.scope_id == scope_id, NODES.c.node_key == node_key(node_id)).values(**values)
        )


def topic_visible(scope_id, artifact_id):
    """An unregistered Topic stays visible; any available alias supports it."""
    managed = select(NODES.c.node_key).where(
        NODES.c.scope_id == scope_id, NODES.c.kind == "topic", NODES.c.artifact_id == artifact_id
    )
    return ~managed.exists() | managed.where(NODES.c.available.is_(True)).exists()


async def blocked_topics(connection: AsyncConnection, scope_id: str) -> frozenset[str]:
    rows = await connection.scalars(
        select(NODES.c.artifact_id)
        .where(NODES.c.scope_id == scope_id, NODES.c.kind == "topic")
        .group_by(NODES.c.artifact_id)
        .having(func.max(NODES.c.available).is_(False))
    )
    return frozenset(str(row) for row in rows)
