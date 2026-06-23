"""Shared state for the unified FastAPI app (memory graph + event loop)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import aiosqlite

    from backend.agents.my_graph import MemoryGraph

graph: Optional["MemoryGraph"] = None
conn: Optional["aiosqlite.Connection"] = None
main_loop: Optional[asyncio.AbstractEventLoop] = None
