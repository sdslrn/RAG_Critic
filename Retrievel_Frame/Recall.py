from concurrent.futures import ThreadPoolExecutor, as_completed
from Retrievel_Frame.faiss_index import *
import torch
from utils.utils import load_json_file, write_json_file, format_prompt
from tqdm import tqdm
from fuzzywuzzy import fuzz
from utils.rank_chunk_by_selfrag_response import rank_chunk_by_selfrag_response
from vllm import LLM, SamplingParams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

retrieval_tasks = [
    "dragon_plus",
    "contriever_msmarco",
    "dpr",
    'bge',
    'e5',
    'gte',
    'contriever',
    'mpnet',
    'minilm'
]

retrieval_class = {
    "dragon_plus": DragonPlusRetriever,
    "contriever_msmarco": ContrieverMsmarcoRetriever,
    "dpr": DprRetriever,
    'bge': BgeRetriever,
    'e5': E5Retriever,
    'gte': GteRetriever,
    'contriever': ContrieverRetriever,
    'mpnet': MpNetRetriever,
    'minilm': MiniLmRetriever
}

data_config = {
    "dragon_plus": {
        "model_path": "/home/dongsheng/Model/dragon-plus-query-encoder",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/dragon_plus/medical_with_answer/*",
    },
    "contriever_msmarco": {
        "model_path": "/home/dongsheng/Model/contriever-msmarco",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/contriever_msmarco/medical_with_answer/*",
    },
    "dpr": {
        "model_path": "/home/dongsheng/Model/dpr-question_encoder-single-nq-base",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/dpr/medical_with_answer/*",
    },
    'bge': {
        "model_path": "/home/dongsheng/Model/bge-base-en/",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/bge/medical_with_answer/*",
    },
    'e5': {
        "model_path": "/home/dongsheng/Model/e5-base-v2/",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/e5/medical_with_answer/*",
    },
    'gte': {
        "model_path": "/home/dongsheng/Model/gte-base/",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/gte/medical_with_answer/*",
    },
    'contriever': {
        "model_path": "/home/dongsheng/Model/contriever/",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/contriever/medical_with_answer/*",
    },
    'mpnet': {
        "model_path": "/home/dongsheng/Model/mpnet-base-nli/",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/mpnet/medical_with_answer/*",
    },
    'minilm': {
        "model_path": "/home/dongsheng/Model/all-MiniLM-L6-v2/",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/minilm/medical_with_answer/*",
        "vector_size": 384
    }
}

dataset_paths = [
    "data/datasets/processed/medwiki.json",
    "data/datasets/processed/medquad.json",
]


def process_task(task, dataset_path):
    dataset_data = load_json_file(dataset_path)
    ret = retrieval_class[task](**data_config[task])
    result_list = []
    for item in tqdm(dataset_data):
        if is_test:
            if item["id"] > 10:
                break
        docs = ret.search_document(item["query"], 10000)  # 10000为Faiss最大的检索数量
        rank = 10000
        for doc in docs:
            if fuzz.token_set_ratio(doc['text'], item['answer']) > 90:  # 90为相似度阈值（满分100）
                rank = docs.index(doc)
                break
        result_list.append({
            "id": item["id"],
            "rank": rank
        })
    filename = dataset_path.split('/')[-1]  # 提取文件名，即 'medwiki.json'
    dataset_name = filename.split('.')[0]  # 去掉扩展名，得到 'medwiki'
    save_path = f"data/results/{dataset_name}_{task}.json"
    write_json_file(save_path, result_list)
    print(f"Finished retrieving for {task}")


is_test = False
already_recalled = [
    "dragon_plus",
    "contriever_msmarco",
    "dpr",
    'bge',
    'e5',
    'gte',
]

with ThreadPoolExecutor() as executor:
    futures = []
    for dataset_path in dataset_paths:
        for task in retrieval_tasks:
            if task in already_recalled:
                continue
            futures.append(executor.submit(process_task, task, dataset_path))

    for future in as_completed(futures):
        future.result()
