import requests

from langchain.embeddings.base import Embeddings


class ClovaEmbeddings(Embeddings):
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    def _call_api(self, texts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "texts": texts,
        }

        res = requests.post(self.endpoint, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["embeddings"]

    def embed_documents(self, texts):
        return self._call_api(texts)

    def embed_query(self, text):
        return self._call_api([text])[0]
