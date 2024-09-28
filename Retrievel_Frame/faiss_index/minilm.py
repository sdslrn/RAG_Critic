from .base_retrieval import BaseRetriever
import torch


class MiniLmRetriever(BaseRetriever):
    def __init__(self, model_path: str, device: str, passages, passages_embeddings, vector_size):
        super().__init__(model_path, device, passages, passages_embeddings, vector_size)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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
                    sentence_embeddings = self.mean_pooling(model_output, encoded_batch['attention_mask'])
                    embeddings.append(sentence_embeddings.cpu())
                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)

        return embeddings.numpy()
