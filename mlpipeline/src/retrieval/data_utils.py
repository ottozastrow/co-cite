import os
import json
import tqdm

import pandas as pd
# import datasets
# from transformers import AutoTokenizer
import wandb
from haystack.nodes import PreProcessor

# import parse_retrieval_data
# import config
import cocitedata


def reinsert_citations(data):
    """data contains text with @cit@ tags. Replace them with citations"""
    res = ""
    segments = data["txt"].split("@cit@")
     
    for i in range(len(segments)-1):
        try:
            res += segments[i]
            res += data["citation_texts"][i]
        except:
            print("skipping segment")
            print(len(segments) -1, len(data["citation_texts"]))
    res += segments[-1]
    
    return res


def setup_preprocessor(args):
    preprocessor = PreProcessor(
        clean_empty_lines=False,
        clean_whitespace=False,
        split_respect_sentence_boundary=False,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
    )
    return preprocessor

def read_docs_batch_from_json(batch):
    """read a batch of json files and return a list of documents"""
    docs = []
    for file in batch:
        with open(file) as f:
            data = json.load(f)
            fulltext = reinsert_citations(data)
            doc = {'content': fulltext, 'meta': {'bva_id': data['bva_id']}}
            docs.append(doc)
    return docs


def load_retrieval_dataset(args) -> tuple[list, list]:
    ######## load datasets #########
    all_filepaths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json')]
    if args.samples != -1:
        all_filepaths = all_filepaths[:args.samples]
    if args.debug:
        all_filepaths = all_filepaths[:10]
    
    test_split = 0.1 if not args.debug else 0.5
    test_split_index = round(len(all_filepaths) * (1-test_split))

    train_filepaths = all_filepaths[:test_split_index]
    test_filepaths = all_filepaths[test_split_index:]

    assert len(test_filepaths) > 0
    return train_filepaths, test_filepaths


def load_labels(args, filepaths):
    """json data is the format provided by the BVA dataset.
    we convert from one row per document to one row per citation."""

    print("reading json files into df")
    # create batches of files
    filebatchsize=100
    batches = [filepaths[i:i+filebatchsize] for i in range(0, len(filepaths), filebatchsize)]
    questions = []
    answers = []
    # contexts = []
    for i in tqdm.tqdm(range(len(batches))):
        df = None
        batch = batches[i]
        for file in batch:
            all_data = pd.read_json(file, lines=True) # read data frame from json file
            data = cocitedata.flatten_df(all_data, args)  # expensive operation
            if df is None:
                df = data
            else:
                df = pd.concat([df, data], copy=False, ignore_index=True)
        
            # with open(file) as f:
            #     data = json.load(f)
            #     fulltext = reinsert_citations(data)
            #     # for every row in df add the same context
            #     for i in range(len(df)):
            #         contexts.append(fulltext)            

        questions.extend(df["text"].to_list())
        answers.extend(df["label"].to_list())

    return questions, answers