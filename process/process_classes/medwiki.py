from process.process_classes.base_dataset import BaseDataset
from utils.utils import load_json_file, write_json_file
import os


class MedWiki(BaseDataset):
    def __init__(self, data_path: str, save_path: str):
        super().__init__()
        self.data_path = data_path
        self.save_path = save_path
        self.data = self.preprocess()

    def load_data_row(self):
        for filename in os.listdir(self.data_path):
            if filename.endswith('.json'):
                real_path = self.data_path + '/' + filename
                row_data = load_json_file(real_path)
                return row_data

    def preprocess(self):
        row_data = self.load_data_row()
        processed_data = []
        for i, item in enumerate(row_data):
            new_item = {
                "id": i,
                "query": item.get("input"),
                "answer": item.get("output")
            }
            processed_data.append(new_item)
        return processed_data

    def save_data(self):
        write_json_file(self.save_path, self.data)
