from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader, PreProcessor, DensePassageRetriever, RAGenerator, BM25Retriever, SentenceTransformersRanker

from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from haystack.utils import launch_es

import config
import cocitedata

import tqdm
import os
import json
import numpy as np
import pandas as pd
import random


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


def build_document_store(args, document_store, filepaths):
    launch_es()
    if args.rebuild_dataset:
        document_store.delete_all_documents()

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )

    filebatchsize=100
    batches = [filepaths[i:i+filebatchsize] for i in range(0, len(filepaths), filebatchsize)]
    
    for i in tqdm.tqdm(range(len(batches))):
        docs = []
        batch = batches[i]
        for file in batch:

            with open(file) as f:
                data = json.load(f)
                fulltext = reinsert_citations(data)
                doc = {'content': fulltext, 'meta': {'bva_id': data['bva_id']}}
                docs.append(doc)
        docs_processed = preprocessor.process(docs)
        document_store.write_documents(docs_processed)
    
    return document_store


def load_questions(args, filepaths):

    """json data is the format provided by the BVA dataset.
    This function converts it to parquet format.
    
    beforer converting to parquet the data is processed
    we convert from one row per document to one row per citation."""


    print("reading json files into df")
    # create batches of files
    filebatchsize=100
    batches = [filepaths[i:i+filebatchsize] for i in range(0, len(filepaths), filebatchsize)]
    questions = []
    answers = []
    for i in tqdm.tqdm(range(len(batches))):
        df = None
        batch = batches[i]
        for file in batch:
            data = pd.read_json(file, lines=True) # read data frame from json file
            data = cocitedata.preprocess_data(data, args)  # expensive operation
            if df is None:
                df = data
            else:
                df = pd.concat([df, data], copy=False, ignore_index=True)
            
        questions.extend(df["text"].to_list())
        answers.extend(df["label"].to_list())

    return questions, answers


def docs_contain_citation(docs, citation):
    """Check if any of the documents contains the citation
    if so return the index of the document containing the citation
    else return -1"""

    for i in range(len(docs)):
        if citation in docs[i].content:
            doc_index = i
            return doc_index
    return -1


def print_metrics(mrr):
    # count values that arent -1
    positives = [x for x in mrr if x != -1]
    print("Positives:", positives)
    print("Total:", len(mrr))
    negatives = len(mrr) - len(positives)
    print("Negatives:", negatives)
    print("Accuracy:", len(positives) / len(mrr))

    # average vaulue of positives
    print("Average MRR:", np.mean(positives))


def main():
    args = config.cmd_arguments()
    # document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
    document_store = InMemoryDocumentStore()
    all_filepaths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json')]
    if args.samples != -1:
        all_filepaths = all_filepaths[:args.samples]
    
    test_split = 0.1
    test_split_index = round(len(all_filepaths) * (1-test_split))

    kb_filepaths = all_filepaths[:test_split_index]
    test_filepaths = all_filepaths[test_split_index:]
    if args.rebuild_dataset:
        document_store = build_document_store(args, document_store, kb_filepaths)

    if args.retriever == "bm25":
        retriever = BM25Retriever(document_store)
    elif args.retriever == "dpr":
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=True,
            embed_title=True,
        )
    else:
        raise ValueError("Unknown retriever")

    retriever.debug = True
    if args.rebuild_dataset:
        if args.retriever != "bm25":
            document_store.update_embeddings(retriever)

    # Initialize RAG Generator
    # generator = RAGenerator(
    #     model_name_or_path="facebook/rag-token-nq",
    #     use_gpu=True,
    #     top_k=3,
    #     max_length=200,
    #     min_length=2,
    #     # embed_title=True,
    #     num_beams=3,
    # )
    questions, answers = load_questions(args, test_filepaths)
    mrr = []
    # shuffle the order of questions and answers
    both = list(zip(questions, answers))
    random.seed(42)
    random.shuffle(both)
    questions, answers = zip(*both)

    for i in tqdm.tqdm(range(len(questions))):
        question = questions[i]
        retrieved_docs = retriever.retrieve(query=question, top_k=50)
        mrr.append(docs_contain_citation(retrieved_docs, answers[i]))
        if i%100 == 0:
            print_metrics(mrr)
        # ranked_docs = ranker.predict(query=question, documents=retrieved_docs)
        # gen_answers = generator.predict(query=question, documents=ranked_docs)
        # print(gen_answers)
        # res = pipe.run(query=question, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
        # print_answers(s, details="minimum")
    print_metrics(mrr)


if __name__ == "__main__":
    main()