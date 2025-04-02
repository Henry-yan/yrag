import os
from typing import List
import glob


class DocumentLoader:
    """
    文档加载器，用于从指定目录加载文档
    """
    def __init__(self, documents_dir: str):
        self.documents_dir = documents_dir
    
    def load_documents(self, file_pattern: str = "*") -> List[dict]:
        """
        加载文档并返回文档列表
        :param file_pattern: 文件匹配模式，默认为所有文件
        :return: 文档列表，每个文档是一个包含内容和元数据的字典
        """
        documents = []
        path_pattern = os.path.join(self.documents_dir, file_pattern)
        
        for file_path in glob.glob(path_pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # 创建文档对象
                    document = {
                        'content': content,
                        'metadata': {
                            'source': file_path,
                            'filename': os.path.basename(file_path)
                        }
                    }
                    documents.append(document)
            except Exception as e:
                print(f"加载文档 {file_path} 时出错: {e}")
        
        return documents