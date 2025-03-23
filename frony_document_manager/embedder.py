import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
load_dotenv()

class HuggingFaceEmbedder():
    def __init__(self, model_id: str, embed_dim: int, batch_size: int = 4):
        self.model = SentenceTransformer(model_id, device=os.getenv('EMBEDDING_MODEL_DEVICE', "cpu"))
        self.embed_dim = embed_dim
        self.batch_size = batch_size

    def embed(self, data: str | List[str]):
        return [item[:self.embed_dim] for item in self.model.encode([data] if isinstance(data, str) else data, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True).half().tolist()]

class OpenAIEmbedder():
    def __init__(self, model_id: str, embed_dim: int):
        self.client = OpenAI()
        self.model_id = model_id
        self.embed_dim = embed_dim

    @staticmethod
    def normalize_embeddings(embeddings: List[List[float]], p: float = 2) -> List[List[float]]:
        return [(item / np.linalg.norm(item, ord=p)).tolist() for item in np.array(embeddings)]

    def embed(self, data: str | List[str]):
        response = self.client.embeddings.create(
            input=[data] if isinstance(data, str) else data,
            model=self.model_id,
            dimensions=self.embed_dim,
        )
        return self.normalize_embeddings([item.embedding for item in response.data])
