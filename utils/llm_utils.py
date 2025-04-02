import requests
import json
from typing import List, Dict, Any, Optional


class BaseLLM:
    """
    大模型基类
    """
    def generate(self, prompt: str, context: Optional[List[str]] = None) -> str:
        """
        生成回答
        :param prompt: 用户问题
        :param context: 上下文信息
        :return: 生成的回答
        """
        raise NotImplementedError("子类必须实现此方法")


class OllamaLLM(BaseLLM):
    """
    使用Ollama调用本地大模型
    """
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
    
    def generate(self, prompt: str, context: Optional[List[str]] = None) -> str:
        """
        使用Ollama生成回答
        :param prompt: 用户问题
        :param context: 检索到的上下文文档
        :return: 生成的回答
        """
        url = f"{self.base_url}/api/generate"
        
        # 构建系统提示
        system_prompt = "你是一个有帮助的AI助手。请根据给定的上下文信息回答用户的问题。"
        
        # 添加上下文
        if context and len(context) > 0:
            context_text = "\n\n".join(context)
            full_prompt = f"{system_prompt}\n\n上下文信息:\n{context_text}\n\n用户问题: {prompt}\n\n回答:"
        else:
            full_prompt = f"{system_prompt}\n\n用户问题: {prompt}\n\n回答:"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "无法生成回答")
        except Exception as e:
            print(f"Ollama调用失败: {e}")
            return f"Ollama调用出错: {str(e)}"


class OpenAILLM(BaseLLM):
    """
    使用OpenAI API调用大模型
    """
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
    
    def generate(self, prompt: str, context: Optional[List[str]] = None) -> str:
        """
        使用OpenAI API生成回答
        :param prompt: 用户问题
        :param context: 检索到的上下文文档
        :return: 生成的回答
        """
        import openai
        
        openai.api_key = self.api_key
        
        messages = []
        
        # 系统消息
        messages.append({
            "role": "system",
            "content": "你是一个有帮助的AI助手。请根据给定的上下文信息回答用户的问题。"
        })
        
        # 添加上下文
        if context and len(context) > 0:
            context_text = "\n\n".join(context)
            messages.append({
                "role": "user",
                "content": f"上下文信息:\n{context_text}"
            })
        
        # 添加用户问题
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return f"OpenAI API调用出错: {str(e)}"