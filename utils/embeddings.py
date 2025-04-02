import requests
import numpy as np
from typing import List, Dict, Any, Optional


class BaseEmbeddings:
    """
    向量嵌入基类
    """
    def embed_text(self, text: str) -> List[float]:
        """
        将单个文本转换为向量
        :param text: 输入文本
        :return: 向量表示
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def embed_documents(self, documents: List[dict]) -> List[dict]:
        """
        向量化文档列表
        :param documents: 文档列表
        :return: 包含向量的文档列表
        """
        embedded_documents = []
        for doc in documents:
            embedding = self.embed_text(doc['content'])
            embedded_doc = {
                **doc,
                'embedding': embedding
            }
            embedded_documents.append(embedded_doc)
        
        return embedded_documents


class SentenceTransformersEmbeddings(BaseEmbeddings):
    """
    使用sentence-transformers的向量嵌入
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
    
    def embed_text(self, text: str) -> List[float]:
        """
        将单个文本转换为向量
        :param text: 输入文本
        :return: 向量表示
        """
        return self.model.encode(text).tolist()
    
    def embed_documents(self, documents: List[dict]) -> List[dict]:
        """
        向量化文档列表
        :param documents: 文档列表
        :return: 包含向量的文档列表
        """
        texts = [doc['content'] for doc in documents]
        embeddings = self.model.encode(texts)
        
        embedded_documents = []
        for i, doc in enumerate(documents):
            embedded_doc = {
                **doc,
                'embedding': embeddings[i].tolist()
            }
            embedded_documents.append(embedded_doc)
        
        return embedded_documents


class OpenAIEmbeddings(BaseEmbeddings):
    """
    使用OpenAI API的向量嵌入
    """
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def embed_text(self, text: str) -> List[float]:
        """
        使用OpenAI API将文本转换为向量
        :param text: 输入文本
        :return: 向量表示
        """
        try:
            payload = {
                "input": text,
                "model": self.model_name
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            embedding = result['data'][0]['embedding']
            
            return embedding
        except Exception as e:
            print(f"OpenAI嵌入模型调用失败: {e}")
            # 返回一个空向量作为后备
            return [0.0] * 1536  # OpenAI embedding-ada-002 长度为1536
    
    def embed_documents(self, documents: List[dict]) -> List[dict]:
        """
        向量化文档列表
        :param documents: 文档列表
        :return: 包含向量的文档列表
        """
        # 对于较大规模的文档集，应考虑分批处理以避免API限制
        # 这里为简单起见，直接使用基类的逐个处理方法
        return super().embed_documents(documents)


class OllamaEmbeddings(BaseEmbeddings):
    """
    使用Ollama的向量嵌入
    """
    def __init__(self, base_url: str, model_name: str = "nomic-embed-text"):
        self.base_url = base_url
        self.model_name = model_name
        self.api_url = f"{base_url}/api/embeddings"
    
    def embed_text(self, text: str) -> List[float]:
        """
        使用Ollama将文本转换为向量
        :param text: 输入文本
        :return: 向量表示
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get('embedding', [])
            
            return embedding
        except Exception as e:
            print(f"Ollama嵌入模型调用失败: {e}")
            # 返回一个空向量作为后备，具体长度取决于Ollama使用的模型
            return [0.0] * 384  # 典型的embedding模型长度
    
    def embed_documents(self, documents: List[dict]) -> List[dict]:
        """
        向量化文档列表
        :param documents: 文档列表
        :return: 包含向量的文档列表
        """
        # Ollama可能不支持批量嵌入，使用基类的逐个处理方法
        return super().embed_documents(documents)


class Embeddings:
    """
    嵌入工厂类，根据配置选择合适的嵌入模型
    """
    @staticmethod
    def create(model_type: str, **kwargs) -> BaseEmbeddings:
        """
        创建嵌入模型实例
        :param model_type: 模型类型，可选值: "sentence-transformers", "openai", "ollama"
        :param kwargs: 其他参数
        :return: 嵌入模型实例
        """
        if model_type == "sentence-transformers":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            return SentenceTransformersEmbeddings(model_name)
        
        elif model_type == "openai":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("OpenAI嵌入模型需要API密钥")
            
            model_name = kwargs.get("model_name", "text-embedding-ada-002")
            return OpenAIEmbeddings(api_key, model_name)
        
        elif model_type == "ollama":
            base_url = kwargs.get("base_url", "http://localhost:11434")
            model_name = kwargs.get("model_name", "nomic-embed-text")
            return OllamaEmbeddings(base_url, model_name)
        
        else:
            raise ValueError(f"不支持的嵌入模型类型: {model_type}")