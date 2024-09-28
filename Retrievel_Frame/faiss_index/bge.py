from .base_retrieval import BaseRetriever
import torch


class BgeRetriever(BaseRetriever):
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
                    model_output = self.model(**encoded_batch)
                    sentence_embeddings = model_output[0][:, 0]
                    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                    embeddings.append(sentence_embeddings.cpu())
                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)

        return embeddings.numpy()
