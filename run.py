from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from tensorboardX import SummaryWriter

from load_dataset import load_fold_dataset
from preprocessing import build_train_loader, build_test_loader, get_token_list
from espi_model import ESPI_MSG_MODEL
from eval_model import calc_metrics
from config import Config


def test_loader_on_model(test_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds, gts, logits = [], [], []
    for data in tqdm(test_loader, desc="Testing"):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)

        pred = torch.round(out).long().tolist()
        gt = data.y.tolist()

        logits.extend(out.tolist())
        preds.extend(pred)
        gts.extend(gt)

    acc, prc, rec, f1 = calc_metrics(preds, gts)
    return (acc, prc, rec, f1), (logits, gts)


def train(
    train_data,
    num_epochs,
    output_path,
    log_filename="",
    log_step=10,
    train_data_filename="train_data",
    test_data_filename="test_data",
    test_data=None,
    test_step=100,
):
    train_loader = build_train_loader(train_data, output_path, train_data_filename)
    test_loader = (
        build_test_loader(test_data, output_path, test_data_filename)
        if test_data
        else None
    )

    token_list = get_token_list(output_path)
    token_num = len(token_list)

    model = ESPI_MSG_MODEL(
        hidden_size=128, layer_num=2, token_num=token_num, dropout=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    log_filename += datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(output_path / "runs" / log_filename)

    model.train()
    logger.info("Start training")

    global_step = 0
    for epoch in range(num_epochs):
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            global_step += 1

            data = data.to(device)

            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)

            loss = criterion(out, data.y.float())

            loss.backward()
            optimizer.step()

            if global_step % log_step == 0:
                writer.add_scalar("Loss/train", loss, global_step)

            if test_loader and global_step % test_step == 0:
                acc, pre, rec, f1 = test_loader_on_model(test_loader, model)[0]

                writer.add_scalar("Accuracy/test", acc, global_step)
                writer.add_scalar("Precision/test", pre, global_step)
                writer.add_scalar("Recall/test", rec, global_step)
                writer.add_scalar("F1/test", f1, global_step)

    writer.close()

    model_path = output_path / "model.pt"
    torch.save(model, model_path)

    return model


def test(test_data, output_path, test_data_filename="test_data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = output_path / "model.pt"
    model = torch.load(model_path).to(device)

    test_loader = build_test_loader(test_data, output_path, test_data_filename)

    logger.info("Start testing")

    return test_loader_on_model(test_loader, model)


def main():
    train_data, test_data = load_fold_dataset("5_fold_datasets", "spidb", 0)

    output_dir_test = Config.output_dir / "tmp"
    output_dir_test.mkdir(exist_ok=True, parents=True)

    model = train(
        train_data=train_data,
        num_epochs=30,
        output_path=output_dir_test,
        test_data=test_data,
        test_step=500,
    )

    test(test_data=test_data, output_path=output_dir_test)


if __name__ == "__main__":
    main()
