from .base_retrieval import BaseRetriever
import torch


class ContrieverRetriever(BaseRetriever):
    def __init__(self, model_path: str, device: str, passages, passages_embeddings):
        super().__init__(model_path, device, passages, passages_embeddings)

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
                    query_emb = self.model(**encoded_batch)
                    token_embeddings = query_emb[0]
                    mask = encoded_batch['attention_mask']
                    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
                    query_emb = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                    embeddings.append(query_emb.cpu())
                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings.numpy()
