"""
Unified FastAPI: ingestion (A) + memory/decision (B), single process.

Run from repository root:
    set PYTHONPATH=.
    python main.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend import app_state
from backend.app.api.my_router import router as input_processing_router
from backend.agents.my_graph import MemoryGraph
from backend.routers import decision, ingestion
from backend.services.meeting_memory_service import ALLOWED_MSGPACK_MODULES, ShortTermMemoryStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from backend.utils.path_tool import get_abs_path

logger = logging.getLogger("ggd")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.main_loop = asyncio.get_running_loop()
    asyncio.create_task(ingestion.process_events())

    store = ShortTermMemoryStore()
    serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
    app_state.conn = await aiosqlite.connect(get_abs_path(store.DB_URI))
    saver = AsyncSqliteSaver(app_state.conn, serde=serde)
    app_state.graph = MemoryGraph(saver)
    logger.info("MemoryGraph initialized (SQLite checkpoint ready)")

    def _warmup_ocr():
        try:
            from backend.legacy.extract_speaker_num import preload_ocr

            preload_ocr()
        except Exception as e:
            logger.warning("OCR pre-load failed (will retry lazily): %s", e)

    threading.Thread(target=_warmup_ocr, daemon=True).start()

    yield

    if app_state.conn:
        await app_state.conn.close()
    app_state.graph = None
    app_state.conn = None
    app_state.main_loop = None
    logger.info("Connection closed")


app = FastAPI(
    title="GGD Unified Ingestion + Memory API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion.router)
app.include_router(decision.router)
app.include_router(input_processing_router, prefix="/api/v1", tags=["input-processing"])


if __name__ == "__main__":
    import uvicorn

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="127.0.0.1", port=9888, log_level="info")
