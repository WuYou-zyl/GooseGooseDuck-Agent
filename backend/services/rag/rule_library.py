import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.model.factory import get_embedding_model
from backend.utils.config_handler import chroma_conf
from backend.utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    text_loader,
    xlsx_loader,
)
from backend.utils.logger_handler import logger
from backend.utils.path_tool import get_abs_path
from langchain_core.documents import Document


class RuleLibrary(object):
    def __init__(self, embedding=None):
        embedding = embedding or get_embedding_model()
        self.rl = chroma_conf["rule_library"]
        self.vec_store = Chroma(
            collection_name=self.rl["collection_name"],
            embedding_function=embedding,
            persist_directory=get_abs_path(self.rl["persist_directory"]),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rl["chunk_size"],
            chunk_overlap=self.rl["chunk_overlap"],
            separators=self.rl["separators"],
        )
        self.load_document() # 每次启动时加载文档

    def get_retriever(self):
        return self.vec_store.as_retriever(search_kwargs={"k": self.rl["k"]})

    def load_document(self):
        def check_md5_hex(md5_hex: str) -> int:
            store_path = get_abs_path(self.rl["md5_hex_store"])
            if not os.path.exists(store_path):
                open(store_path, "w", encoding="utf-8").close()
                return 0
            with open(store_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_hex:
                        return 1
            return 2

        def save_md5_hex(md5_hex: str) -> None:
            with open(get_abs_path(self.rl["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_hex + "\n")

        def get_file_documents(read_path: str) -> list[Document]:
            ext = os.path.splitext(read_path)[1].lower()
            if ext == ".pdf":
                return pdf_loader(read_path)
            if ext == ".txt":
                return text_loader(read_path)
            if ext == ".xlsx":
                return xlsx_loader(read_path)
            return []

        allowed_files_paths = listdir_with_allowed_type(
            get_abs_path(self.rl["data_path"]),
            tuple(self.rl["allow_knowledge_files_type"]),
        )

        for path in allowed_files_paths:
            md5_hex = get_file_md5_hex(path)
            if md5_hex is None:
                continue
            if check_md5_hex(md5_hex) == 1:
                logger.info(f"Document {path} has been loaded.")
                continue
            elif check_md5_hex(md5_hex) == 2:
                # 需要根据根据 metadata 中的路径值删除此前加载的文档
                col = self.vec_store._collection  # LangChain Chroma 暴露的底层 collection
                res = col.get(where={"source": path})  # 若你用的 Chroma 要求，则 where 写成 {"source": {"$eq": path}}
                n = len(res["ids"])
                self.vec_store.delete(where={"source": path})
                logger.info(f"Document {path} {n} documents has been deleted.")

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"Document {path} is empty.")
                    continue

                split_document: list[Document] = self.splitter.split_documents(documents)
                if not split_document:
                    logger.warning(f"Document {path} is empty after split.")
                    continue

                self.vec_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"Document {path} has been loaded.")
            except Exception:
                logger.error(f"Load document {path} failed.", exc_info=True)
                continue


if __name__ == "__main__":
    vs = RuleLibrary()
    retriever = vs.get_retriever()
    res = retriever.invoke("超能力者")
    for r in res:
        print(r.page_content)
