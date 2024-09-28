from abc import ABC, abstractmethod
import os
import pickle
import time
import glob

import numpy as np
from transformers import AutoTokenizer, AutoModel

import Retrievel_Frame.faiss_index.index
from utils.utils import load_json_file

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class BaseRetriever(ABC):
    def __init__(self, model_path: str, device, passages, passages_embeddings, vector_size=768):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = device
        self.model.eval()
        self.model = self.model.to(device)
        #  初始化index索引（Faiss的index索引）
        self.index = Retrievel_Frame.faiss_index.index.Indexer(vector_size, 0, 8)

        input_paths = glob.glob(passages_embeddings)
        input_paths = sorted(input_paths)  # 将input_paths按照字母顺序排序
        embeddings_dir = os.path.dirname(input_paths[0])  # os.path.dirname：去掉文件名，返回目录
        index_path = os.path.join(embeddings_dir, "index.faiss")

        if os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            # input_paths包含了各种embeddings文件
            self.index_encoded_data(self.index, input_paths, 1000000)
            print(f"Indexing time: {time.time() - start_time_indexing:.1f} s.")
            self.index.serialize(embeddings_dir)

        print("loading passages")
        self.passages = load_json_file(passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

    @abstractmethod
    def embed_queries(self, queries):
        pass

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []  # embedding所对应的id
        allembeddings = np.array([])  # 所有的embeddings
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)  # ids和embeddings中的"s"代表其为列表

            # 将此文件中的ids数据添加到allids中，将此文件中的embeddings数据添加到allembeddings中
            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            # 如果allembeddings的大小大于indexing_batch_size，则将allembeddings中的数据添加到index中
            # 返回剩余的allembeddings和allids，继续和下一个文件中的ids和embeddings数据进行拼接
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        # 将剩余的ids和allembeddings中的数据添加到index中
        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        print("Data indexing completed.")

    @staticmethod
    def add_embeddings(index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        # ids_toadd和embeddings_toadd为从ids和embeddings中取出的数据
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        # ids和embeddings为剩余的数据
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        # 将ids_toadd和embeddings_toadd添加到index中
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    @staticmethod
    def add_passages(passages, top_passages_and_scores):
        # add passages to original data
        # top_passages_and_scores[0][0]即为db_ids
        # passages为id到具体文本的映射
        docs = [passages[doc_id] for doc_id in top_passages_and_scores[0][0]]
        return docs

    def search_document(self, query, n_docs=10):
        # 将query转换为embeddings（这里只有一个query）
        questions_embedding = self.embed_queries([query])

        # get top k results
        start_time_retrieval = time.time()
        # top_ids_and_scores：[
        #   [db_ids, scores],  # db_ids为前n_docs个id，scores为前n_docs个id对应的分数, db_ids和scores的长度为n_docs，都是列表
        #   [db_ids, scores],  # 由于只有一个query，所以top_ids_and_scores只有上面一行，此行其实不存在
        #   ...
        # ]
        top_ids_and_scores = self.index.search_knn(questions_embedding, n_docs)
        # print(f"Search time: {time.time() - start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:n_docs]
