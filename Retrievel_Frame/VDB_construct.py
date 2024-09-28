from embedding_classes import *
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

embedding_model_name = [
    "dragon_plus",
    "contriever_msmarco",
    "dpr",
    "bge",
    "e5",
    "gte",
    "contriever",
    "mpnet",
    "minilm"
]

passage_name = [
    "medical_with_answer"
]

data_config = {
    embedding_model_name[0] + "_" + passage_name[0]: {
        "model_path": "/home/dongsheng/Model/dragon-plus-context-encoder",
        "passages_path": "data/corpus/processed/medical_with_answer.json",
        "device": device,
        "save_path_prefix": f"data/corpus/vector_database/{embedding_model_name[0]}/{passage_name[0]}",
        "is_test": False
    },
    embedding_model_name[1] + "_" + passage_name[0]: {
        "model_path": "/home/dongsheng/Model/contriever-msmarco",
        "passages_path": "data/corpus/processed/medical_with_answer.json",
        "device": device,
        "save_path_prefix": f"data/corpus/vector_database/{embedding_model_name[1]}/{passage_name[0]}",
        "is_test": False
    },
    embedding_model_name[2] + "_" + passage_name[0]: {
        "model_path": "/home/dongsheng/Model/dpr-ctx_encoder-single-nq-base",
        "passages_path": "data/corpus/processed/medical_with_answer.json",
        "device": device,
        "save_path_prefix": f"data/corpus/vector_database/{embedding_model_name[2]}/{passage_name[0]}",
        "is_test": False
    },
    embedding_model_name[3] + "_" + passage_name[0]: {
        "model_path": "/home/dongsheng/Model/bge-base-en/",
        "passages_path": "data/corpus/processed/medical_with_answer.json",
        "device": device,
        "save_path_prefix": f"data/corpus/vector_database/{embedding_model_name[3]}/{passage_name[0]}",
        "is_test": False
    },
    embedding_model_name[4] + "_" + passage_name[0]: {
        "model_path": "/home/dongsheng/Model/e5-base-v2/",
        "passages_path": "data/corpus/processed/medical_with_answer.json",
        "device": device,
        "save_path_prefix": f"data/corpus/vector_database/{embedding_model_name[4]}/{passage_name[0]}",
        "is_test": False
    },
    embedding_model_name[5] + "_" + passage_name[0]: {
        "model_path": "/home/dongsheng/Model/gte-base/",
        "passages_path": "data/corpus/processed/medical_with_answer.json",
        "device": device,
        "save_path_prefix": f"data/corpus/vector_database/{embedding_model_name[5]}/{passage_name[0]}",
        "is_test": False
    },
    embedding_model_name[6] + "_" + passage_name[0]: {
        "model_path": "/home/dongsheng/Model/contriever",
        "passages_path": "data/corpus/processed/medical_with_answer.json",
        "device": device,
        "save_path_prefix": f"data/corpus/vector_database/{embedding_model_name[6]}/{passage_name[0]}",
        "is_test": False
    },
    embedding_model_name[7] + "_" + passage_name[0]: {
        "model_path": "/home/dongsheng/Model/mpnet-base-nli",
        "passages_path": "data/corpus/processed/medical_with_answer.json",
        "device": device,
        "save_path_prefix": f"data/corpus/vector_database/{embedding_model_name[7]}/{passage_name[0]}",
        "is_test": False
    },
    embedding_model_name[8] + "_" + passage_name[0]: {
        "model_path": "/home/dongsheng/Model/all-MiniLM-L6-v2",
        "passages_path": "data/corpus/processed/medical_with_answer.json",
        "device": device,
        "save_path_prefix": f"data/corpus/vector_database/{embedding_model_name[8]}/{passage_name[0]}",
        "is_test": False
    }
}

class_config = {
    embedding_model_name[0] + "_" + passage_name[0]: DragonPlus,
    embedding_model_name[1] + "_" + passage_name[0]: ContrieverMsmarco,
    embedding_model_name[2] + "_" + passage_name[0]: Dpr,
    embedding_model_name[3] + "_" + passage_name[0]: Bge,
    embedding_model_name[4] + "_" + passage_name[0]: E5,
    embedding_model_name[5] + "_" + passage_name[0]: Gte,
    embedding_model_name[6] + "_" + passage_name[0]: Contriever,
    embedding_model_name[7] + "_" + passage_name[0]: MpNet,
    embedding_model_name[8] + "_" + passage_name[0]: MiniLm
}

already_constructed = [
    embedding_model_name[0] + "_" + passage_name[0],
    embedding_model_name[1] + "_" + passage_name[0],
    embedding_model_name[2] + "_" + passage_name[0],
    embedding_model_name[3] + "_" + passage_name[0],
    embedding_model_name[4] + "_" + passage_name[0],
    embedding_model_name[5] + "_" + passage_name[0],
]


def construct_vdb(vdb_name, config):
    print(f"Constructing {vdb_name}...")
    vdb = class_config[vdb_name](**config)
    vdb.vdb_constructor()
    print(f"{vdb_name} constructed.\n")


# 使用多线程进行并行构建
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(construct_vdb, vdb_name, config): vdb_name
        for vdb_name, config in data_config.items()
        if vdb_name not in already_constructed
    }

    for future in as_completed(futures):
        vdb_name = futures[future]
        try:
            future.result()  # 获取线程的返回值或异常
        except Exception as exc:
            print(f"{vdb_name} generated an exception: {exc}")

print("All constructions completed.")