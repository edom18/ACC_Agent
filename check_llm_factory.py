import os
import sys
from acc_agent.llm_factory import get_llm_model, get_embedding_function, LangChainEmbeddingAdapter
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb.utils import embedding_functions

def test_openai_default():
    print("Testing OpenAI Default...")
    os.environ["ACC_LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "dummy"
    if "ACC_LLM_MODEL" in os.environ:
        del os.environ["ACC_LLM_MODEL"]
        
    llm = get_llm_model()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-5-mini"
    print("PASS: Defaults to OA/gpt-5-mini")

def test_gemini_switch():
    print("Testing Gemini Switch...")
    os.environ["ACC_LLM_PROVIDER"] = "gemini"
    os.environ["GOOGLE_API_KEY"] = "dummy"
    
    llm = get_llm_model()
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == "gemini-2.5-flash"
    print("PASS: Switches to Gemini/gemini-2.5-flash")

def test_explicit_model_env():
    print("Testing Explicit Model Env...")
    os.environ["ACC_LLM_PROVIDER"] = "gemini"
    os.environ["ACC_LLM_MODEL"] = "gemini-1.5-flash"
    
    llm = get_llm_model()
    assert llm.model == "gemini-1.5-flash"
    print("PASS: Respects ACC_LLM_MODEL")

def test_explicit_arg():
    print("Testing Explicit Argument...")
    os.environ["ACC_LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "dummy"
    
    llm = get_llm_model(model_name="gpt-3.5-turbo")
    assert llm.model_name == "gpt-3.5-turbo"
    print("PASS: Argument overrides everything")

def test_embedding_openai():
    print("Testing Embedding OpenAI...")
    os.environ["ACC_LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "dummy"
    
    ef = get_embedding_function()
    assert isinstance(ef, embedding_functions.OpenAIEmbeddingFunction)
    print("PASS: Embedding OpenAI")

def test_embedding_gemini():
    print("Testing Embedding Gemini...")
    os.environ["ACC_LLM_PROVIDER"] = "gemini"
    os.environ["GOOGLE_API_KEY"] = "dummy"
    
    ef = get_embedding_function()
    assert isinstance(ef, LangChainEmbeddingAdapter)
    print("PASS: Embedding Gemini (Adapter)")

if __name__ == "__main__":
    try:
        test_openai_default()
        test_gemini_switch()
        test_explicit_model_env()
        test_explicit_arg()
        test_embedding_openai()
        test_embedding_gemini()
        print("\nALL TESTS PASSED")
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
