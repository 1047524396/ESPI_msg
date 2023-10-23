from datasets import load_dataset, Dataset
from typing import Literal
import os


def load_fold_dataset(
    folds_data_dir: str, name: Literal["patchdb", "spidb"], fold: int
) -> tuple[Dataset, Dataset]:
    data_file = os.path.join(folds_data_dir, f"{name}_fold.json")
    data = load_dataset("json", data_files=data_file, field="data", split="train")

    folds = data["fold"]
    train_indices = [i for i, f in enumerate(folds) if f != fold]
    test_indices = [i for i, f in enumerate(folds) if f == fold]

    train_data = data.select(train_indices)
    test_data = data.select(test_indices)

    return train_data, test_data


def load_whole_dataset(
    folds_data_dir: str, name: Literal["patchdb", "spidb"]
) -> Dataset:
    data_file = os.path.join(folds_data_dir, f"{name}_fold.json")

    data = load_dataset("json", data_files=data_file, field="data", split="train")
    return data


if __name__ == "__main__":
    train_data, test_data = load_fold_dataset("5_fold_datasets", "spidb", 0)
    labels = train_data["label"]
    msgs = train_data["msg"]
    #print(train_data[:10])
    print(labels[:10])
    print(msgs[:10])
