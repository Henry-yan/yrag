# 配置文件
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 模型配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI 嵌入模型

# Ollama配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "qwen2:latest"
OLLAMA_EMBEDDING_MODEL = "quentinz/bge-large-zh-v1.5:latest"  # Ollama 嵌入模型，确保已安装此模型

# 嵌入模型配置
EMBEDDING_MODEL_TYPE = "sentence-transformers"  # 可选值: "sentence-transformers", "openai", "ollama"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 使用sentence-transformers中的模型

# 向量存储配置
VECTOR_DB_PATH = "vector_db"  # 向量数据库存储路径

# 文档配置
DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200