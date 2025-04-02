import os
import argparse
from utils.document_loader import DocumentLoader
from utils.text_splitter import TextSplitter
from utils.embeddings import Embeddings
from utils.vector_store import VectorStore
from utils.llm_utils import OllamaLLM, OpenAILLM
import config


class RAGApp:
    """
    RAG应用主类
    """
    def __init__(self, use_ollama_llm: bool = True, embedding_model_type: str = None):
        # 初始化各组件
        self.document_loader = DocumentLoader(config.DOCUMENTS_DIR)
        self.text_splitter = TextSplitter(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        
        # 设置嵌入模型类型
        if embedding_model_type is None:
            embedding_model_type = config.EMBEDDING_MODEL_TYPE
        
        # 根据嵌入模型类型初始化嵌入模型
        if embedding_model_type == "sentence-transformers":
            self.embeddings = Embeddings.create(
                "sentence-transformers", 
                model_name=config.EMBEDDING_MODEL
            )
            print(f"使用Sentence Transformers嵌入模型: {config.EMBEDDING_MODEL}")
        
        elif embedding_model_type == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("未设置OpenAI API密钥，请在.env文件中设置OPENAI_API_KEY")
            
            self.embeddings = Embeddings.create(
                "openai", 
                api_key=config.OPENAI_API_KEY,
                model_name=config.OPENAI_EMBEDDING_MODEL
            )
            print(f"使用OpenAI嵌入模型: {config.OPENAI_EMBEDDING_MODEL}")
        
        elif embedding_model_type == "ollama":
            self.embeddings = Embeddings.create(
                "ollama",
                base_url=config.OLLAMA_BASE_URL,
                model_name=config.OLLAMA_EMBEDDING_MODEL
            )
            print(f"使用Ollama嵌入模型: {config.OLLAMA_EMBEDDING_MODEL}")
        
        else:
            raise ValueError(f"不支持的嵌入模型类型: {embedding_model_type}")
        
        self.vector_store = VectorStore(config.VECTOR_DB_PATH)
        
        # 根据参数选择LLM
        if use_ollama_llm:
            print(f"使用Ollama模型: {config.OLLAMA_MODEL}")
            self.llm = OllamaLLM(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
        else:
            if not config.OPENAI_API_KEY:
                raise ValueError("未设置OpenAI API密钥，请在.env文件中设置OPENAI_API_KEY")
            print(f"使用OpenAI模型: {config.OPENAI_MODEL}")
            self.llm = OpenAILLM(config.OPENAI_API_KEY, config.OPENAI_MODEL)
    
    # [其余方法保持不变]
    def index_documents(self, file_pattern: str = "*"):
        """
        索引文档
        :param file_pattern: 文件匹配模式
        """
        print(f"正在加载文档: {file_pattern}")
        documents = self.document_loader.load_documents(file_pattern)
        print(f"加载了 {len(documents)} 个文档")
        
        if not documents:
            print("没有找到文档，请检查文档目录和文件模式")
            return
        
        print("正在分割文档...")
        chunked_documents = self.text_splitter.split_documents(documents)
        print(f"文档分割为 {len(chunked_documents)} 个块")
        
        print("正在向量化文档...")
        embedded_documents = self.embeddings.embed_documents(chunked_documents)
        
        print("正在添加文档到向量存储...")
        self.vector_store.add_documents(embedded_documents)
        print("文档索引完成")
    
    def answer_question(self, question: str, num_results: int = 4):
        """
        回答问题
        :param question: 用户问题
        :param num_results: 检索的文档数量
        :return: 生成的回答
        """
        print(f"问题: {question}")
        
        # 对问题进行向量化
        print("正在向量化问题...")
        question_embedding = self.embeddings.embed_text(question)
        
        # 检索相关文档
        print(f"正在检索相关文档 (top {num_results})...")
        results = self.vector_store.similarity_search(question_embedding, k=num_results)
        
        if not results:
            print("没有找到相关文档")
            return self.llm.generate(question)
        
        # 提取相关上下文
        contexts = [doc['content'] for doc in results]
        
        # 打印检索到的文档摘要
        print(f"检索到 {len(results)} 个相关文档")
        for i, doc in enumerate(results):
            source = doc['metadata'].get('source', 'unknown')
            print(f"文档 {i+1}: {source}, 得分: {doc.get('score', 0)}")
        
        # 生成回答
        print("正在生成回答...")
        answer = self.llm.generate(question, contexts)
        
        return answer


def main():
    parser = argparse.ArgumentParser(description="RAG应用")
    parser.add_argument('--index', action='store_true', help='索引文档')
    parser.add_argument('--file-pattern', type=str, default="*", help='索引时使用的文件匹配模式')
    parser.add_argument('--ollama', action='store_true', help='使用Ollama LLM，默认为True')
    parser.add_argument('--openai', action='store_true', help='使用OpenAI API LLM')
    parser.add_argument('--embedding-model', type=str, choices=['sentence-transformers', 'openai', 'ollama'],
                       help='使用的嵌入模型类型，默认使用配置文件中的设置')
    parser.add_argument('--question', type=str, help='要回答的问题')
    
    args = parser.parse_args()
    
    # 根据参数确定使用哪个LLM
    use_ollama_llm = not args.openai  # 默认使用Ollama，除非指定--openai
    
    try:
        app = RAGApp(use_ollama_llm=use_ollama_llm, embedding_model_type=args.embedding_model)
        
        # 索引文档
        if args.index:
            app.index_documents(args.file_pattern)
        
        # 回答问题
        if args.question:
            answer = app.answer_question(args.question)
            print("\n回答:")
            print(answer)
        
        # 如果没有指定操作，进入交互模式
        if not args.index and not args.question:
            print("进入交互模式，输入'exit'或'quit'退出")
            while True:
                question = input("\n请输入问题: ")
                if question.lower() in ['exit', 'quit']:
                    break
                
                answer = app.answer_question(question)
                print("\n回答:")
                print(answer)
    
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()