import os
import shutil
import math
import time
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS

KNOWLEDGE_BASE_DIR = "knowledge_base"
VECTOR_DB_DIR = "faiss_index_db"
ZHIPUAI_API_KEY = "2c2a258f6daf4ab1a7eaa1f1298f5a0d.Lajx17gMNJr4bk9F"

if not ZHIPUAI_API_KEY: print("错误：请设置 ZHIPUAI_API_KEY。"); exit()
if not os.path.exists(KNOWLEDGE_BASE_DIR): print(f"错误：文件夹 '{KNOWLEDGE_BASE_DIR}' 不存在。"); exit()

def load_documents(directory_path):
    """加载指定目录下的所有支持的文档。"""
    print(f"正在从 '{directory_path}' 加载文档...")
    docs = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"  - 正在加载: {file_path}")
            try:
                if file.endswith(".pdf"): loader = PyPDFLoader(file_path)
                elif file.endswith(".md"): loader = UnstructuredMarkdownLoader(file_path)
                elif file.endswith(".docx"): loader = Docx2txtLoader(file_path)
                elif file.endswith(".txt"): loader = UnstructuredFileLoader(file_path, encoding='utf-8')
                else: print(f"    - 跳过不支持的文件类型: {file}"); continue
                docs.extend(loader.load())
            except Exception as e: print(f"    - 加载文件 {file_path} 时发生错误: {e}")
    print(f"成功加载 {len(docs)} 个文档片段（加载前）。")
    return docs

def build_vector_store():
    """构建并使用 FAISS 持久化向量数据库（分批处理）。"""
    try:
        documents = load_documents(KNOWLEDGE_BASE_DIR)
        if not documents: print("没有加载到任何文档，脚本将退出。"); return

        print("正在切分文档...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"文档被切分为 {len(chunks)} 个块。")

        print("正在初始化 ZhipuAI Embedding 模型...")
        embeddings = ZhipuAIEmbeddings(api_key=ZHIPUAI_API_KEY)

        print(f"正在使用 FAISS 构建向量数据库...")
        if os.path.exists(VECTOR_DB_DIR):
            print(f"发现旧的 FAISS 索引目录，正在删除: {VECTOR_DB_DIR}")
            shutil.rmtree(VECTOR_DB_DIR)

        BATCH_SIZE = 20  # 保持较小的批次大小
        vector_store = None
        total_batches = math.ceil(len(chunks) / BATCH_SIZE)

        for i in range(0, len(chunks), BATCH_SIZE):
            batch_docs = chunks[i:i + BATCH_SIZE]
            current_batch_num = (i // BATCH_SIZE) + 1
            print(f"  - 正在处理批次 {current_batch_num} / {total_batches} ({len(batch_docs)} 个块)...")

            try:
                print(f"    -- 正在调用 Embedding API 并添加到 FAISS (批次 {current_batch_num})...")
                if vector_store is None:
                    # 第一批：创建 FAISS 索引
                    vector_store = FAISS.from_documents(batch_docs, embeddings)
                else:
                    # 后续批次：添加到现有 FAISS 索引
                    vector_store.add_documents(batch_docs)
                print(f"    -- 批次 {current_batch_num} 添加成功。")

            except Exception as e:
                print(f"!!!! 处理批次 {current_batch_num} 时发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                return

            print(f"    -- 批次 {current_batch_num} 处理完毕，暂停 1 秒...")
            time.sleep(1)

        if vector_store:
            print(f"正在将 FAISS 索引保存到 '{VECTOR_DB_DIR}'...")
            vector_store.save_local(VECTOR_DB_DIR) # FAISS 需要显式保存
            print("-" * 30)
            print("索引构建完成！")
            print(f"FAISS 索引已成功保存在 '{VECTOR_DB_DIR}' 文件夹中。")
            print("-" * 30)
        else:
            print("未能创建向量数据库。")

    except Exception as e:
        print(f"!!!!!! 脚本执行过程中发生顶层错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    build_vector_store()