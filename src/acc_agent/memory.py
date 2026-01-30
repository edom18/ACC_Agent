import os
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings

class ArtifactStore:
    """
    外部ストア（External Memory）。
    過去のログやドキュメントを保存し、セマンティック検索を可能にする。
    """
    def __init__(self, collection_name: str = "acc_artifacts"):
        # Use persistent client to store data
        self.client = chromadb.PersistentClient(path="./.chroma_db")
        
        # We need an embedding function. Since we are using OpenAI in this project, 
        # we'll use OpenAI's embeddings via LangChain-compatible wrapper or direct.
        # Direct chromadb OpenAI wrapper is also fine.
        api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_fn
        )

    def add_artifact(self, content: str, metadata: Optional[dict] = None, id: Optional[str] = None):
        """過去のログなどを保存する"""
        import uuid
        self.collection.add(
            documents=[content],
            metadatas=[metadata] if metadata else [{}],
            ids=[id if id else str(uuid.uuid4())]
        )

    def recall(self, query: str, n_results: int = 3) -> List[str]:
        """クエリに基づいて情報を検索する"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

# Test implementation/mock for demonstration if needed
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    store = ArtifactStore()
    store.add_artifact("明日の天気は晴れです。", {"source": "weather_api"})
    print(store.recall("天気はどうですか？"))
