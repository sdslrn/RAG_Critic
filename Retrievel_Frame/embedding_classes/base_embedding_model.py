from abc import ABC, abstractmethod
import os

from typing import List


class BaseEmbeddingModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load_model(self):
        """
        This function is used to load the model.
        """
        pass

    def sentence_to_embedding(self, sentences: List[str]):
        """
        This function is used to convert the sentences to the embeddings.
        """
        pass

    def load_passages(self):
        """
        This function is used to load the passages.
        """
        pass

    def save_embeddings(self, embeddings, save_path: str):
        """
        This function is used to save the embeddings.
        """
        pass

    def vdb_constructor(self):
        """
        This function is used to construct the vdb.
        """
        pass
