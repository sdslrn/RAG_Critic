import json
from utils.utils import load_json_file, write_json_file

if __name__ == "__main__":
    passages_path = "/home/dongsheng/Code/ECNU/RAG_Critic/data/corpus/row/PubMed_PMC_CPG_Textbook.jsonl"
    medwiki_path = "/home/dongsheng/Code/ECNU/RAG_Critic/data/datasets/processed/medwiki.json"
    medquad_path = "/home/dongsheng/Code/ECNU/RAG_Critic/data/datasets/processed/medquad.json"
    save_path = "/home/dongsheng/Code/ECNU/RAG_Critic/data/corpus/processed/medical_with_answer.json"

    # 将PubMed_PMC_CPG_Textbook.jsonl中的text字段和medwiki.json、medquad.json中的answer字段合并到一个文件中
    passages = []
    id = 0
    with open(passages_path) as fin:
        for k, line in enumerate(fin):
            ex = json.loads(line)
            passages.append({
                "id": id,
                "text": ex["text"]
            })
            id += 1

    with open(medwiki_path) as fin:
        medwiki_data = json.load(fin)
    for i, item in enumerate(medwiki_data):
        passages.append({
            "id": id,
            "text": item["answer"]
        })
        id += 1

    with open(medquad_path) as fin:
        medquad_data = json.load(fin)
    for i, item in enumerate(medquad_data):
        passages.append({
            "id": id,
            "text": item["answer"]
        })
        id += 1

    print(len(passages))

    # 将passages写入到medical_with_answer.json文件中
    write_json_file(save_path, passages)

    # 读取medical_with_answer.json文件中的前5条数据
    # data = load_json_file(save_path)
    # print(data[:5])
