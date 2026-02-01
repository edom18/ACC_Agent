import os
from typing import List, Optional
import chromadb
from .llm_factory import get_embedding_function

class ArtifactStore:
    """
    外部ストア（External Memory）。
    過去のログやドキュメントを保存し、セマンティック検索を可能にする。
    """
    def __init__(self, collection_name: str = "acc_artifacts"):
        # Use persistent client to store data
        self.client = chromadb.PersistentClient(path="./.chroma_db")
        
        # Use factory to get the correct embedding function based on provider
        self.embedding_fn = get_embedding_function()
        
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
