from .base_retrieval import BaseRetriever
import torch
from torch import Tensor
import torch.nn.functional as F


class E5Retriever(BaseRetriever):
    def __init__(self, model_path: str, device: str, passages, passages_embeddings):
        super().__init__(model_path, device, passages, passages_embeddings)

    @staticmethod
    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_queries(self, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                batch_question.append(q)
                if len(batch_question) == 16 or k == len(queries) - 1:
                    encoded_batch = self.tokenizer(
                        batch_question,
                        return_tensors="pt",
                        max_length=512,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda(self.device) for k, v in encoded_batch.items()}
                    outputs = self.model(**encoded_batch)
                    embeddings_sentences = self.average_pool(outputs.last_hidden_state, encoded_batch['attention_mask'])
                    embeddings_sentences = F.normalize(embeddings_sentences, p=2, dim=1)
                    embeddings.append(embeddings_sentences.cpu())
                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)

        return embeddings.numpy()
