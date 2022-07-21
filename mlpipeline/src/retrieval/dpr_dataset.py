import os
import pandas as pd
import tqdm
import datasets
from transformers import AutoTokenizer
import wandb

import parse_retrieval_data
import cocitedata
import config


def build_dpr_dataset(sections):
    """builds DPR dataformat from haystack library.
    format per sample from website
        "dataset": str,
        "question": str,
        "answers": list of str
        "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
        "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
        "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    """
    dpr_data = []
    for citation in sections.keys():
        section = sections[citation]
        dpr_data.append({
            "dataset": "uscode",
            "question": section["citation"],
            "answers": [section["title"]],
            "positive_ctxs": [],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        })
    return dpr_data

if __name__ == "__main__":
    args = config.cmd_arguments()

    wandb.init(project="cocite", tags=["DPR"], config=args, mode=args.wandb_mode)

    data, tokenizer = cocitedata.load_dataset(args)

    import pdb
    pdb.set_trace()
