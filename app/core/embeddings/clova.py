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

        payload = {"text": texts}

        res = requests.post(self.endpoint, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["result"]["embedding"]

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            vector = self._call_api(text)
            embeddings.append(vector)
        return embeddings

    def embed_query(self, texts):
        return self._call_api(texts)
