import re
import itertools
from collections import Counter
import pickle as pkl

import torch
from torch_geometric.data import DataLoader, Data
import spacy
from tqdm.contrib.concurrent import process_map
from loguru import logger

from config import Config

""" move to global space to decrease the time of loading model """
nlp = spacy.load("en_core_web_sm")


def embedding(long_text):
    lines = long_text.splitlines()

    tokens, edge_list = [], []

    node_counter = 0
    for line in lines:
        doc = nlp(line)
        tokens.extend([token.text for token in doc])

        """
        token.head: The syntactic parent, or “governor”, of this token.
        token.dep: Syntactic dependency relation.
        """
        for token in doc:
            if token.head is not token:
                edge_list.append([token.i + node_counter, token.head.i + node_counter])

        node_counter += len(doc)

    edge_index = torch.tensor(edge_list).t()
    return tokens, edge_index


def get_token_list(model_dir):
    p = model_dir / "token_dict.pkl"
    if not p.exists():
        logger.error(f"Token dict not found in {p}")
        exit(1)

    token_list = pkl.load(open(p, "rb"))
    logger.info(f"Token dict loaded from {p}, has {len(token_list)} tokens")

    return token_list


def build_data_loader(
    tokens: list[list[str]],
    edges: list[torch.Tensor],
    labels: list[int],
    token_list: list[str],
):
    token2id = {token: i for i, token in enumerate(token_list)}
    all_data = []

    for i, (token, edge, label) in enumerate(zip(tokens, edges, labels)):
        # get token mapping (in case token does not appear in token_list)
        token_ids = []

        if len(token) == 0:
            token.append("<unk>")
        for t in token:
            t = t.lower()
            token_ids.append(token2id[t] if t in token2id else token2id["<unk>"])

        token_ids = torch.LongTensor(token_ids)

        data = Data(x=token_ids, edge_index=edge.long(), y=label)
        all_data.append(data)

    loader = DataLoader(all_data, batch_size=32)
    return loader


def build_train_loader(train_data, model_dir, name="train_data"):
    token_list_path = model_dir / "token_dict.pkl"
    train_data_path = model_dir / f"{name}.pkl"

    if token_list_path.exists() and train_data_path.exists():
        data = pkl.load(open(train_data_path, "rb"))
        token_list = get_token_list(model_dir)
        loader = build_data_loader(token_list=token_list, **data)
        return loader

    msg = [m.strip() for m in train_data["msg"]]
    labels = train_data["label"]

    tokens, edges = list(
        zip(*process_map(embedding, msg, max_workers=Config.max_workers, chunksize=100))
    )

    token_list = list(itertools.chain.from_iterable(tokens))
    # token_list = [t.lower() for t in token_list if not re.match(r"^\s*$", t)]

    # c = Counter(token_list)
    # token_list = list(set([t for t in token_list if c[t] > 1]))
    token_list = list(set(token_list))
    token_list.append("<unk>")
    logger.info(f"Num of Tokens size: {len(token_list)}")

    pkl.dump(token_list, open(token_list_path, "wb"))
    logger.info(f"Token dict saved to {token_list_path}")

    data = {
        "tokens": tokens,
        "edges": edges,
        "labels": labels,
    }
    pkl.dump(data, open(train_data_path, "wb"))
    loader = build_data_loader(token_list=token_list, **data)

    logger.info("Build train loader finished!")

    return loader


def build_test_loader(test_data, model_dir, name="test_data"):
    test_data_path = model_dir / f"{name}.pkl"

    if test_data_path.exists():
        data = pkl.load(open(test_data_path, "rb"))
        token_list = get_token_list(model_dir)
        loader = build_data_loader(token_list=token_list, **data)
        return loader

    token_list = get_token_list(model_dir)

    msg = [m.strip() for m in test_data["msg"]]

    tokens, edges = list(
        zip(*process_map(embedding, msg, max_workers=Config.max_workers, chunksize=100))
    )
    labels = test_data["label"]

    data = {
        "tokens": tokens,
        "edges": edges,
        "labels": labels,
    }
    pkl.dump(data, open(test_data_path, "wb"))

    loader = build_data_loader(token_list=token_list, **data)

    logger.info("Build test loader finished!")

    return loader
