"""
# design decisions
# 1. we don't reinsert citations in context segments. TODO: change this. only benefits implementation time. during inference the retriever shouldn't rely on the QUERY containing inserted citations, however the retrieved contexts may contain them.

# 2. identity mapping not removed in train set.


building a dpr database
we need a lookup table with key=label and value=inputs
then we collect all matches and put them into the positive ctxs field
if we reuse the existing dataset we save code for tokenization, (not caching since we reindex)
but since we will want to change the input data rather  sooner than later thats a time consuming dead end
so instead: build a key value index
for each document extract citations and contexts
normalize keys
do we tokenize all of them? no. i think dpr takes care of that.

where do we start
cocite data has get_citation_context which lies behind flatten_df

output of this script are train/dev jsons that can be read by retrieve.py


builds DPR dataformat from haystack library.
    format per sample from website
        "dataset": str,
        "question": str,
        "answers": list of str
        "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
        "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
        "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
"""

import os
import random
import json
import tqdm
from collections import defaultdict

import pandas as pd
from traitlets import default
# import datasets
# from transformers import AutoTokenizer
import wandb
from haystack.nodes import PreProcessor

# import parse_retrieval_data
import cocitedata
import config
import retrieval.data_utils as data_utils
import citation_normalization


def dpr_format_passage(passage, i):
    passage_dict = {
        "title": "",
        "text": passage,
        "score": 1,
        "title_score": 0,
        "passage_id": i,
    }
    return passage_dict

def sample_to_dpr_format(dpr_dataset: list, texts: list, citation:str, preprocessor, args):

    for text in texts:
        # randomly select 5 texts from texts
        sampled_texts = random.sample(texts, args.num_positives)
        sample = {
                "dataset": "dpr_with_similar_citations",
                "answers": [citation],
                "question": text,
                "positive_ctxs": [dpr_format_passage(t, i) for i, t in enumerate(sampled_texts)],
                "negative_ctxs": [],
                "hard_negative_ctxs": [],
        }
        dpr_dataset.append(sample)

def citation_pairs_from_docs(filepaths: list, args, max_contexts_per_citation: int) -> dict:
    filebatchsize=100
    batches = [filepaths[i:i+filebatchsize] for i in range(0, len(filepaths), filebatchsize)]

    index: dict = {}  # keys are normalized citations
    pbar = tqdm.tqdm(range(len(batches)))
    for i in pbar:
        pbar.set_description(f"Began indexingkl batch nr. {i} to df")
        for file in batches[i]:
            # read data frame from json file
            documents = pd.read_json(file, lines=True)
            data = cocitedata.flatten_df(documents, args)  # expensive operation
            
            if not args.dont_normalize_citations:
                data["label"] = data["label"].apply(
                    lambda citation: citation_normalization.normalize_citation(
                        citation,
                        remove_subsections=True, remove_subsubsections=True))

            chunk_index = data.set_index("label")["text"].to_dict()
            # append lists in chunk index to lists in index
            for key in chunk_index.keys():
                if key in index:
                    if len(index[key]) < max_contexts_per_citation:
                        index[key].append(chunk_index[key])
                else:
                    index[key] = [chunk_index[key]]
    return index
   

def sample_negative_citations(citation, index, args):
    negative_citations = []
    # get num_negatives random citations
    negative_citations = random.sample(index.keys(), args.num_negatives + 1)
    # +1 one extra incase we randomly included the correct citation
    # if correct citation in negatives drop it 
    negatives = [x for x in negative_citations if x != citation]
    # otherwise drop the one extra citation
    if len(negatives) > args.num_negatives:
        negatives = negatives[:args.num_negatives]

    selected_negative_contexts = []
    for i, negative_citation in enumerate(negative_citations):
        # pick a random context from negative["positive_ctxs"]
        negative_context = random.choice(index[negative_citation])
        selected_negative_contexts.append(dpr_format_passage(negative_context, i))
    return selected_negative_contexts


def read_files(filepaths: list, preprocessor) -> list:
    filebatchsize=100
    batches = [filepaths[i:i+filebatchsize] for i in range(0, len(filepaths), filebatchsize)]
    
    all_docs = []
    for i in tqdm.tqdm(range(len(batches))):
        docs = data_utils.read_docs_batch_from_json(batches[i])
        docs = preprocessor.preprocess_docs(docs)
        all_docs.extend(docs)
    return all_docs


def dpr_dataset_from_citation_pairs(index, preprocessor, args):
    # iterate through index and convert to dpr format
    dpr_dataset = []
    for (citation, texts) in tqdm.tqdm(index.items()):
        sample_to_dpr_format(dpr_dataset, texts, citation, preprocessor, args)

    # drop all samples with less than n positive contexts (especially since one is the identity mapping)
    filtered_dpr_dataset = [
        sample for sample in dpr_dataset
        if len(sample["positive_ctxs"]) >= args.num_positives
    ]
    
    # add negatives
    for sample in filtered_dpr_dataset:
        citation = sample["answers"][0]
        sample["hard_negative_ctxs"] = sample_negative_citations(citation, index, args)
    return filtered_dpr_dataset


def build_dpr(args, doc_dir):
    data_utils.load_retrieval_dataset(args)
    args.num_negatives = 5
    args.num_positives = 1
    max_contexts_per_citation = 50
    dataset_descriptor = {
        "name": "dpr_with_similar_citations",
        "num_negatives": args.num_negatives,
        "num_positives": args.num_positives,
        "max_contexts_per_citation": max_contexts_per_citation,
        "samples": args.samples,
        "dont_normalize_citations": args.dont_normalize_citations,
        "input_tokens": args.input_tokens,
    }

    preprocessor = data_utils.setup_preprocessor(args)

    train_files, test_files = data_utils.load_retrieval_dataset(args)

    test_index = citation_pairs_from_docs(test_files, args, max_contexts_per_citation=max_contexts_per_citation)
    test_dpr_dataset = dpr_dataset_from_citation_pairs(test_index, preprocessor, args)
    assert len(test_dpr_dataset) != 0, "No test samples found"

    train_index = citation_pairs_from_docs(train_files, args, max_contexts_per_citation=max_contexts_per_citation)
    train_dpr_dataset = dpr_dataset_from_citation_pairs(train_index, preprocessor, args)
    assert len(train_dpr_dataset) != 0, "No train samples found"

    dataset_descriptor["train_size"] = len(train_dpr_dataset)
    dataset_descriptor["test_size"] = len(test_dpr_dataset)

    print("done building dpr dataset, number of samples for train is:",
        len(train_dpr_dataset), "and test: ", len(test_dpr_dataset))

    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)

    # save dict to pretty printed json
    with open(f"{doc_dir}/train_dpr_dataset.json", "w") as f:
        json.dump(train_dpr_dataset, f, indent=4)
    with open(f"{doc_dir}/test_dpr_dataset.json", "w") as f:
        json.dump(test_dpr_dataset, f, indent=4)

    print("done saving dpr dataset")
    
    with open(f"{doc_dir}/dataset_descriptor.json", "w") as f:
        json.dump(dataset_descriptor, f, indent=4)
    print("done saving dataset descriptor")
