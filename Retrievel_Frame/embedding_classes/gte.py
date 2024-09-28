from typing import List
import numpy as np
from Retrievel_Frame.embedding_classes.base_embedding_model import BaseEmbeddingModel
from utils.utils import load_json_file
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from torch import Tensor
import torch.nn.functional as F


class Gte(BaseEmbeddingModel):
    def __init__(self, model_path: str, passages_path: str, device, save_path_prefix: str, is_test: bool = False):
        super().__init__()
        self.model_path = model_path
        self.passages_path = passages_path
        self.device = device
        self.save_path_prefix = save_path_prefix
        self.tokenizer, self.model = self.load_model()
        self.passages = self.load_passages()
        self.is_test = is_test

    def load_model(self):
        print(f"Loading model from {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModel.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        print("running on", self.device)
        return tokenizer, model

    def load_passages(self):
        print(f"Loading passages from {self.passages_path}")
        return load_json_file(self.passages_path)

    @staticmethod
    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def sentence_to_embedding(self, sentences: List[str]):
        """
        This function is used to convert the sentences to the embeddings.
        :param sentences:
        :return: embeddings
        """
        with torch.no_grad():
            batch_dict = self.tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # (Optionally) normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu()

    def save_embeddings(self, infos, save_path: str):
        # print(f"Saving embeddings to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(infos, f)

    def vdb_constructor(self, batch_size=4):
        """
        This function is used to construct the vdb with batch size = 4.
        """
        embeddings = []
        ids = []
        embedding_file_id = 0
        batch_passages = []

        if self.is_test:
            print("Running in test mode. Only processing the first 100 passages.")
            self.passages = self.passages[:200]

        for k, passage in tqdm(enumerate(self.passages), total=len(self.passages)):
            batch_passages.append(passage.get("text"))
            ids.append(passage.get("id"))

            # Process the batch when it reaches the batch size
            if len(batch_passages) == batch_size:
                batch_embeddings = self.sentence_to_embedding(batch_passages)
                # torch.cuda.empty_cache()
                embeddings.append(batch_embeddings)
                batch_passages = []

            # Save to file every 20,000 passages
            if (k + 1) % 20000 == 0:
                embeddings = torch.cat(embeddings, dim=0)
                embeddings = embeddings.numpy()
                save_path = f"{self.save_path_prefix}/passages_embeddings{embedding_file_id}.pkl"
                self.save_embeddings((ids, embeddings), save_path)
                embeddings = []
                ids = []
                embedding_file_id += 1

        # Process remaining passages
        if batch_passages:
            batch_embeddings = self.sentence_to_embedding(batch_passages)
            torch.cuda.empty_cache()
            embeddings.append(batch_embeddings)

        # Save the last batch of embeddings
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.numpy()
        save_path = f"{self.save_path_prefix}/passages_embeddings{embedding_file_id}.pkl"
        self.save_embeddings((ids, embeddings), save_path)

