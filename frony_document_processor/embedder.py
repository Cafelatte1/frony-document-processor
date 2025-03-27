import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
from torch.cuda import is_available
load_dotenv()

class SentenceTransformerEmbedder():
    def __init__(self, model_id: str, embed_dim: int, batch_size: int = 4, device: str = "cuda" if is_available() else "cpu", precision: str = "fp16"):
        self.model = SentenceTransformer(model_id, device=device)
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.precision = precision

    def embed(self, data: str | List[str]):
        data = [data] if isinstance(data, str) else data
        sorted = np.array([len(self.model.tokenizer.tokenize(i)) for i in data]).argsort()
        embed = self.model.encode(data[sorted], batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True)
        embed[sorted] = embed.half().copy() if self.precision == "fp16" else embed.copy()
        embed = embed.tolist()
        return embed

class OpenAIEmbedder():
    def __init__(self, model_id: str, embed_dim: int):
        self.client = OpenAI()
        self.model_id = model_id
        self.embed_dim = embed_dim

    @staticmethod
    def normalize_embeddings(embeddings: List[List[float]], p: float = 2) -> List[List[float]]:
        return [(item / np.linalg.norm(item, ord=p)).tolist() for item in embeddings]

    def embed(self, data: str | List[str]):
        data = [data] if isinstance(data, str) else data
        response = self.client.embeddings.create(
            input=data,
            model=self.model_id,
            dimensions=self.embed_dim,
        )
        embed = self.normalize_embeddings([item.embedding for item in response.data])
        return embed
