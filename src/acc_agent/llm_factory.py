import os
from typing import Optional, List, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from chromadb import EmbeddingFunction
from chromadb.utils import embedding_functions

class LangChainEmbeddingAdapter(EmbeddingFunction):
    """
    LangChainのEmbeddingsクラスをChromaDBのEmbeddingFunctionプロトコルに適合させるアダプター。
    """
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embeddings_model.embed_documents(input)

def get_embedding_function() -> EmbeddingFunction:
    """
    環境変数 ACC_LLM_PROVIDER に基づいて、ChromaDB用のEmbeddingFunctionを返します。
    """
    provider = os.getenv("ACC_LLM_PROVIDER", "openai").lower()
    
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY is not set for Gemini provider.")
            
        # langchain-google-genai の Embeddings を使用して Adapter でくるむ
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", # 最新の埋め込みモデル
            google_api_key=api_key
        )
        return LangChainEmbeddingAdapter(embeddings)
        
    else: # openai
        api_key = os.getenv("OPENAI_API_KEY")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )


def get_llm_model(model_name: Optional[str] = None, temperature: float = 0.7) -> BaseChatModel:
    """
    環境変数 ACC_LLM_PROVIDER に基づいて、適切なLLMモデルインスタンスを返します。
    
    Args:
        model_name (Optional[str]): モデル名を明示的に指定する場合に使用。
                                    指定がない場合は環境変数 ACC_LLM_MODEL またはデフォルト値を使用。
        temperature (float): 温度パラメータ。

    Returns:
        BaseChatModel: LangChainのChatModelインスタンス (ChatOpenAI or ChatGoogleGenerativeAI)
    """
    provider = os.getenv("ACC_LLM_PROVIDER", "openai").lower()
    
    # モデル名の決定優先順位: 引数 > 環境変数 > デフォルト
    if not model_name:
        model_name = os.getenv("ACC_LLM_MODEL")

    if provider == "gemini":
        if not model_name:
            model_name = "gemini-2.5-flash" # Defauit for Gemini if not set
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # ユーザーへの警告ログなどを出すべきだが、ここではエラーになるに任せるか、あるいはメッセージを出す
            print("Warning: GOOGLE_API_KEY is not set for Gemini provider.")

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True # Gemini sometimes needs this for system prompts in older versions, but useful to keep safety
        )
        
    else: # default to openai
        if not model_name:
            model_name = "gpt-5-mini" # Default for OpenAI if not set
            
        return ChatOpenAI(
            model=model_name, 
            temperature=temperature
        )
