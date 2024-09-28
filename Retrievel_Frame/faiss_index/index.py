import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm


class Indexer(object):
    
    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8):
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        #self.index_id_to_db_id = np.empty((0), dtype=np.int64)
        self.index_id_to_db_id = []

    def index_data(self, ids, embeddings):  # ids: List, embeddings: np.array
        self._update_id_mapping(ids)  # 更新id映射，self.index_id_to_db_id为列表，下标为index_id，值为db_id
        embeddings = embeddings.astype('float32')  # 转换数据类型到float32（faiss仅支持float32） 
        if not self.index.is_trained:  # 如果index没有被训练过
            self.index.train(embeddings)  # 训练index
        self.index.add(embeddings)  # 添加数据到index

        print(f'Total data indexed {len(self.index_id_to_db_id)}')  # 打印index的大小

    # 一种并行查询的方法，并行度为index_batch_size
    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048) -> List[Tuple[List[object], List[float]]]:  # query_vectors: np.array, top_docs: int, index_batch_size: int = 2048
        query_vectors = query_vectors.astype('float32')  # 转换数据类型到float32（faiss仅支持float32）
        result = []  # 结果
        nbatch = (len(query_vectors)-1) // index_batch_size + 1  # 计算迭代次数
        for k in range(nbatch):
            start_idx = k*index_batch_size  # 计算起始索引 
            end_idx = min((k+1)*index_batch_size, len(query_vectors))  # 计算结束索引
            q = query_vectors[start_idx: end_idx]  # 获取一个批次的查询向量
            scores, indexes = self.index.search(q, top_docs)  # 查询
            # convert to external ids
            db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]  # 将内部id转换为外部id
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])  # 添加结果
        return result  # 返回结果

    def serialize(self, dir_path):  # dir_path: str
        index_file = os.path.join(dir_path, 'index.faiss')  # 索引文件
        meta_file = os.path.join(dir_path, 'index_meta.faiss')  # 元数据文件
        print(f'Serializing index to {index_file}, meta data to {meta_file}')  # 打印保存数据信息

        faiss.write_index(self.index, index_file)  # 将index保存到文件
        with open(meta_file, mode='wb') as f:  # 将index_id_to_db_id保存到文件
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')  # 索引文件
        meta_file = os.path.join(dir_path, 'index_meta.faiss')  # 元数据文件
        print(f'Loading index from {index_file}, meta data from {meta_file}')  # 打印加载数据信息

        self.index = faiss.read_index(index_file)  # 从文件加载index
        print('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)  # 打印加载的index信息

        with open(meta_file, "rb") as reader:  # 从文件加载index_id_to_db_id
            self.index_id_to_db_id = pickle.load(reader)
        assert len(  # 检查index_id_to_db_id的大小是否与index的大小一致
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        #new_ids = np.array(db_ids, dtype=np.int64)
        #self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)
        self.index_id_to_db_id.extend(db_ids)  # 更新index_id_to_db_id
