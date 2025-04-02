import os
import json
import numpy as np
from typing import List, Dict, Any
import faiss


class VectorStore:
    """
    向量存储，使用FAISS实现
    """
    def __init__(self, db_path: str = "vector_db"):
        self.db_path = db_path
        self.metadata_path = os.path.join(db_path, "metadata.json")
        self.index = None
        self.documents = []
        
        # 创建数据库目录
        os.makedirs(db_path, exist_ok=True)
        
        # 尝试加载现有索引
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """
        加载现有索引或创建新索引
        """
        index_path = os.path.join(self.db_path, "index.faiss")
        
        if os.path.exists(index_path) and os.path.exists(self.metadata_path):
            try:
                # 加载FAISS索引
                self.index = faiss.read_index(index_path)
                
                # 加载文档元数据
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                
                print(f"已加载向量索引，包含 {len(self.documents)} 个文档")
            except Exception as e:
                print(f"加载索引失败: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """
        创建新索引
        """
        self.documents = []
        self.index = None
        print("创建新的向量索引")
    
    def add_documents(self, documents: List[dict]):
        """
        添加文档到向量存储
        :param documents: 包含向量的文档列表
        """
        if not documents:
            return
        
        vectors = [np.array(doc['embedding'], dtype=np.float32) for doc in documents]
        vectors_matrix = np.vstack(vectors)
        
        # 存储文档（不含向量，以节省空间）
        docs_for_storage = []
        for doc in documents:
            doc_copy = {k: v for k, v in doc.items() if k != 'embedding'}
            docs_for_storage.append(doc_copy)
        
        # 首次添加文档时，创建索引
        if self.index is None:
            dimension = len(vectors[0])
            self.index = faiss.IndexFlatL2(dimension)
        
        # 添加向量到索引
        self.index.add(vectors_matrix)
        
        # 保存文档元数据
        self.documents.extend(docs_for_storage)
        
        # 保存索引和元数据
        self._save_index()
    
    def _save_index(self):
        """
        保存索引和元数据
        """
        index_path = os.path.join(self.db_path, "index.faiss")
        
        # 保存FAISS索引
        faiss.write_index(self.index, index_path)
        
        # 保存文档元数据
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def similarity_search(self, query_vector: List[float], k: int = 4) -> List[dict]:
        """
        执行相似度搜索
        :param query_vector: 查询向量
        :param k: 返回最相似的k个结果
        :return: 检索到的文档
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 确保k不大于索引中的向量数量
        k = min(k, self.index.ntotal)
        
        # 执行搜索
        query_vector_np = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_vector_np, k)
        
        # 获取对应的文档
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                doc['score'] = float(distances[0][i])
                results.append(doc)
        
        return results