from process_classes import *

datasets = [
    "medwiki",
    "medquad"
]

class_config = {
    datasets[0]: MedWiki,
    datasets[1]: MedQuad
}

data_config = {
    datasets[0]: {
        "data_path": "data/datasets/row/medwiki",
        "save_path": "data/datasets/processed/medwiki.json"
    },
    datasets[1]: {
        "data_path": "data/datasets/row/medquad",
        "save_path": "data/datasets/processed/medquad.json"
    }
}


def process(dataset_list: list = None):
    for dataset in dataset_list:
        if dataset not in data_config or dataset not in class_config:
            print(f"Error! Dataset {dataset} not in data_config or class_config")
            continue
        dataset = class_config[dataset](**data_config[dataset])
        dataset.save_data()
        print(f"Dataset {dataset} processed!")


if __name__ == "__main__":
    process(datasets)
