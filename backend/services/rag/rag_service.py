"""
总结服务类。用户提问，搜索资料，获取答案。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from backend.services.rag.rule_library import RuleLibrary
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from backend.model.factory import get_rag_chat_model
from backend.utils.logger_handler import logger
from backend.utils.prompt_loader import load_rag_prompts


class RagSummarizeService(object):
    def __init__(self, rule_library: Optional[RuleLibrary] = None):
        self.rule_library = rule_library or RuleLibrary()
        self.retriever = self.rule_library.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_rag_chat_model()
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | self.debug_prompt | self.model | StrOutputParser()
    
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    async def aretriever_docs(self, query: str) -> list[Document]:
        try:
            return await self.retriever.ainvoke(query)
        except (AttributeError, NotImplementedError, TypeError):
            return await asyncio.to_thread(self.retriever.invoke, query)

    def _build_context(self, context_docs: list[Document]) -> str:
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[参考资料{counter}]：内容：{doc.page_content} | 参考元数据： {doc.metadata}\n"
        return context

    async def arag_summarize(self, query: str) -> str:
        context_docs = await self.aretriever_docs(query)
        context = self._build_context(context_docs)
        payload = {"input": query, "context": context}
        try:
            return await self.chain.ainvoke(payload)
        except (AttributeError, NotImplementedError, TypeError):
            return await asyncio.to_thread(self.chain.invoke, payload)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = self._build_context(context_docs)
        return self.chain.invoke({"input": query, "context": context})
    
    def debug_prompt(self, prompt: str) -> str:
        # logger.debug(f"RAG summarize prompt: {prompt}")
        return prompt


if __name__ == "__main__":
    async def _demo() -> None:
        rag = RagSummarizeService()
        print(await rag.arag_summarize("无责刀"))

    asyncio.run(_demo())