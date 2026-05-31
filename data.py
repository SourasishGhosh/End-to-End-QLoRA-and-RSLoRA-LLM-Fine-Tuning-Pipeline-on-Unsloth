from datasets import Dataset
import json

def get_dataset():
    data_list = []

    print("......loading dataset.....")
    with open("./dataset/dataset.jsonl","r",encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))

    return Dataset.from_list(data_list)