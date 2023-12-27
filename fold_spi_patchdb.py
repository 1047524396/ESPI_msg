from pathlib import Path
import json

from run import train, test
from loguru import logger

import load_dataset
from config import Config


def save_result(result_filepath: Path, name: str, result: dict):
    results = (
        json.loads(result_filepath.read_text()) if result_filepath.exists() else {}
    )

    if name in results:
        logger.error(f"Result {name} already exists, overwrite!")

    results[name] = result

    result_filepath.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    result_base_dir = Path("../result/spi_patchdb")
    result_base_dir.mkdir(exist_ok=True, parents=True)
    result_filepath = result_base_dir / "metric_espi.json"
    # logits_filepath = result_base_dir / "our_dataset_logits_espi.json"

    dataset_names = {"spidb", "patchdb"}
    folds_data_dir = "5_fold_datasets"
    fold_num = 5

    for train_dataset_name in ["patchdb", "spidb"]:
        cross_dataset_name = (dataset_names - {train_dataset_name}).pop()
        for fold_idx in range(fold_num):
            fold_name = f"{train_dataset_name}_fold_{fold_idx}"
            fold_output_dir = Config.output_dir / fold_name
            fold_output_dir.mkdir(exist_ok=True, parents=True)

            train_dataset, test_dataset = load_dataset.load_fold_dataset(
                folds_data_dir, train_dataset_name, fold_idx
            )
            cross_dataset = load_dataset.load_whole_dataset(
                folds_data_dir, cross_dataset_name
            )

            model = train(
                train_data=train_dataset,
                num_epochs=30,
                output_path=fold_output_dir,
                log_filename=fold_name,
                test_data=test_dataset,
                test_data_filename="same",
                test_step=500,
            )

            (acc, pre, rec, f1), (logits, gts) = test(
                test_data=test_dataset,
                output_path=fold_output_dir,
                test_data_filename="same",
            )
            same_result = {
                "accuracy": acc,
                "precision": pre,
                "recall": rec,
                "f1": f1,
            }

            (acc, pre, rec, f1), (logits, gts) = test(
                test_data=cross_dataset,
                output_path=fold_output_dir,
                test_data_filename="cross",
            )
            cross_result = {
                "accuracy": acc,
                "precision": pre,
                "recall": rec,
                "f1": f1,
            }

            save_result(result_filepath, fold_name, {
                'same': same_result,
                'cross': cross_result
            })

            # result = {
            #     "gt": gts,
            #     "logits": logits,
            # }
            # save_result(logits_filepath, fold_name, result)

            logger.info(f"Fold {fold_name} finished!")
