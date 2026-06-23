import aiosqlite

from contextlib import asynccontextmanager

# from langgraph.checkpoint.redis.aio import AsyncRedisSaver
# from langgraph.checkpoint.postgresql.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.model.factory import get_embedding_model

from backend.utils.config_handler import chroma_conf
from backend.utils.path_tool import get_abs_path
from backend.utils.logger_handler import logger
from backend.utils.config_handler import short_memory_conf

from backend.schemas.contract import MemorySummary

# 注册自定义类型，消除 checkpoint 反序列化时的警告
ALLOWED_MSGPACK_MODULES = [
    ("backend.schemas.contract", "IngestionOutput"),
    ("backend.schemas.contract", "PlayerState"),
    ("backend.schemas.contract", "MemorySummary"),
    ("backend.schemas.decision", "DecisionResult"),
    ("backend.schemas.decision", "RuleCriticReview"),
    ("backend.schemas.graph_state", "MemoryDecisionState"),
    ("backend.schemas.graph_state", "SituationSketch"),
    ("backend.schemas.graph_state", "GameSettings"),
    ("backend.schemas.graph_state", "PlayerRosterEntry"),
    ("backend.schemas.graph_state", "EliminationRecord"),
    ("backend.schemas.graph_state", "VoteRecord"),
    ("backend.schemas.graph_state", "MeetingRoundState"),
]

class ShortTermMemoryStore:
    def __init__(self):
        self.DB_URI = short_memory_conf['DB_URL']
    
    @asynccontextmanager
    async def get_saver(self):
        """返回带自定义 serde 的 checkpointer，支持 IngestionOutput/MemorySummary 的序列化"""
        serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
        async with aiosqlite.connect(get_abs_path(self.DB_URI)) as conn:
            yield AsyncSqliteSaver(conn, serde=serde)
        
class LongTermMemoryStore:
    """跨局记忆向量存储，复用现有 Chroma + Embedding。可以先不用"""
    def __init__(self, embedding=None):
        """embedding:嵌入模型的传入"""
        embedding = embedding or get_embedding_model()
        self.vec_store = Chroma(
            collection_name=chroma_conf['long_term_memory']['collection_name'],
            embedding_function=embedding,
            persist_directory=chroma_conf['long_term_memory']['persist_directory']
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf['long_term_memory']['chunk_size'],
            chunk_overlap=chroma_conf['long_term_memory']['chunk_overlap'],
            separators=chroma_conf['long_term_memory']['separators']
        )

    def add(self, doc: MemorySummary) -> None:
        """单条写入。"""
        # TODO: 将 MemorySummary 转为 Document 存入 Chroma
        
    def get_recent(self, session_id: str, k: int = 10) -> list[Document]:
        """按 session_id 获取最近 k 条。需 metadata 过滤或按 sequence_id 排序。"""
        
    def search(self, session_id: str, query: str, k: int = 5) -> list[Document]:
        """语义检索，限定 session_id。"""