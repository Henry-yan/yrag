from typing import List


class TextSplitter:
    """
    文本分块器，将长文本分割成适合嵌入的小块
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成块
        :param text: 输入文本
        :return: 文本块列表
        """
        # 简单的按字符数量分块策略
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # 如果不是最后一块，尝试在一个合适的位置（如句号、换行符）切分
            if end < len(text):
                # 查找可能的分割点
                for split_char in ['\n\n', '\n', '. ', '。', '! ', '！', '? ', '？']:
                    last_split = text[start:end].rfind(split_char)
                    if last_split != -1:
                        end = start + last_split + len(split_char)
                        break
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap  # 重叠部分
            
            # 防止无限循环
            if start >= len(text) or start >= end:
                break
        
        return chunks
    
    def split_documents(self, documents: List[dict]) -> List[dict]:
        """
        将文档列表分割成块
        :param documents: 文档列表
        :return: 分块后的文档列表
        """
        chunked_documents = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc['metadata']
            
            chunks = self.split_text(content)
            
            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    'content': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_id': i
                    }
                }
                chunked_documents.append(chunked_doc)
        
        return chunked_documents