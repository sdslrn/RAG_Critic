from Retrievel_Frame.faiss_index import *
import torch
from utils.utils import load_json_file, write_json_file, format_prompt
from tqdm import tqdm
from fuzzywuzzy import fuzz
from utils.rank_chunk_by_selfrag_response import rank_chunk_by_selfrag_response
from vllm import LLM, SamplingParams

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Load self_rag model ...")
model_path = "/home/dongsheng/Model/self-rag"
model = LLM(model_path, dtype="half")
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

retrieval_tasks = [
    "dragon_plus_self_rag",
]

retrieval_class = {
    "dragon_plus_self_rag": DragonPlusRetriever,
}

data_config = {
    "dragon_plus_self_rag": {
        "model_path": "/home/dongsheng/Model/dragon-plus-query-encoder",
        "device": device,
        "passages": "data/corpus/processed/medical_with_answer.json",
        "passages_embeddings": "data/corpus/vector_database/dragon_plus/medical_with_answer/*",
    },
}

dataset_paths = [
    "data/datasets/processed/medwiki.json",
    "data/datasets/processed/medquad.json",
]

is_test = False
final_chunk_num = 10  # 经过selfrag评定排序后，最终保留的chunk数量

for dataset_path in dataset_paths:
    dataset_data = load_json_file(dataset_path)
    for task in retrieval_tasks:
        print(f"Retrieving for {task}...")
        ret = retrieval_class[task](**data_config[task])
        result_list = []
        for item in tqdm(dataset_data):
            if is_test:
                if item["id"] > 10:
                    break
            # 使用ret根据item["query"]找到item["answer"]的排名
            docs = ret.search_document(item["query"], 10000)  # 10000为Faiss最大的检索数量
            rank = 10000

            prompts = [format_prompt(item["query"], doc["text"]) for doc in docs[:final_chunk_num]]
            # print("len(prompts):", len(prompts))
            preds = model.generate(prompts, sampling_params)
            responses = [pred.outputs[0].text for pred in preds]
            # print("len(responses):", len(responses))
            sequences = rank_chunk_by_selfrag_response(responses)
            final_retrieved_documents = []
            for i in range(final_chunk_num):
                final_retrieved_documents.append(docs[sequences[i]])
            # 将docs前final_chunk_num个文档按照rank的顺序重新排序
            for i in range(final_chunk_num):
                docs[i] = final_retrieved_documents[i]

            for doc in docs:
                if fuzz.token_set_ratio(doc['text'], item['answer']) > 90:  # 90为相似度阈值（满分100）
                    rank = docs.index(doc)
                    break
            result_list.append({
                "id": item["id"],
                "rank": rank
            })
        # 保存结果
        filename = dataset_path.split('/')[-1]  # 提取文件名，即 'medwiki.json'
        dataset_name = filename.split('.')[0]  # 去掉扩展名，得到 'medwiki'
        save_path = f"data/results/{dataset_name}_{task}.json"
        write_json_file(save_path, result_list)
