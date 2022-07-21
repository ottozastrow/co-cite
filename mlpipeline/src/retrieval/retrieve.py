from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader, PreProcessor, DensePassageRetriever, RAGenerator, BM25Retriever, EmbeddingRetriever, SentenceTransformersRanker

from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from haystack.utils import launch_es
from haystack.pipelines import Pipeline
from haystack.nodes import Docs2Answers

import config
import cocitedata

import tqdm
import os
import json
import numpy as np
import pandas as pd
import random

import wandb

def manual_es_launch():
    import os
    from subprocess import Popen, PIPE, STDOUT

    # check if files exist
    if not os.path.exists("elasticsearch-7.9.2-linux-x86_64.tar.gz"):
        import subprocess
        subprocess.call(['wget', 'https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz', '-q']) 
        subprocess.call(['tar', '-xzf', 'elasticsearch-7.9.2-linux-x86_64.tar.gz'])
        subprocess.call(['chown', '-R', 'daemon:daemon', 'elasticsearch-7.9.2'])


    es_server = Popen(
        ["elasticsearch-7.9.2/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1)  # as daemon
    )
    print("starting sleeping")
    # wait until ES has started
    subprocess.call(["sleep", "30"])
    print("done sleeping")


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


def build_document_store(args, document_store, filepaths, preprocessor):

    if args.rebuild_dataset:
        document_store.delete_all_documents()

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


def load_labels(args, filepaths):

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
    contexts = []
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
        
            with open(file) as f:
                data = json.load(f)
                fulltext = reinsert_citations(data)
                # for every row in df add the same context
                for i in range(len(df)):
                    contexts.append(fulltext)            

        questions.extend(df["text"].to_list())
        answers.extend(df["label"].to_list())
        # import pdb
        # pdb.set_trace()

    return questions, answers, contexts


def docs_contain_citation(docs, citation):
    """Check if any of the documents contains the citation
    if so return the index of the document containing the citation
    else return -1"""

    for i in range(len(docs)):
        if citation in docs[i].content:
            doc_index = i
            return doc_index
    return -1


def print_metrics(mrr, k_list):
    recalls = recall_at_k(mrr, k_list)
    print("Recall@k: ", recalls)
    
    # count values that arent -1
    positives = [x for x in mrr if x != -1]
    print("Positives:", positives)
    print("Total:", len(mrr))
    negatives = len(mrr) - len(positives)
    print("Negatives:", negatives)
    print("Accuracy:", len(positives) / len(mrr))

    # average vaulue of positives
    avg_mrr =  np.mean(positives)
    print("Average MRR:", avg_mrr)

    # log metrics in wandb
    wandb.log({"MRR": avg_mrr})
    wandb.log({"Accuracy": len(positives) / len(mrr)})
    wandb.log({"Positives": len(positives)})
    wandb.log({"Negatives": negatives})


def main():
    args = config.cmd_arguments()
    wandb.init(project="cocite", tags=["retrieve"], config=args, mode=args.wandb_mode)

    ######## setup document store #########
    doc_index = "document"
    label_index = "labels"
    use_es_store = True
    if use_es_store:
        document_store = ElasticsearchDocumentStore(
            host="localhost",
            username="",
            password="",
            index=doc_index,
            label_index=label_index,
            similarity="dot_product",
            embedding_dim=768
        )
    else:
        document_store = InMemoryDocumentStore()
    
    
    ######## load datasets #########
    all_filepaths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json')]
    if args.samples != -1:
        all_filepaths = all_filepaths[:args.samples]
    
    test_split = 0.1
    test_split_index = round(len(all_filepaths) * (1-test_split))

    kb_filepaths = all_filepaths[:test_split_index]
    test_filepaths = all_filepaths[test_split_index:]
    assert len(test_filepaths) > 0
    preprocessor = setup_preprocessor(args)
    if args.rebuild_dataset:
        document_store = build_document_store(
            args, document_store, kb_filepaths, preprocessor
        )

    ######## setup retriever #########
    if args.retriever == "bm25":
        retriever = BM25Retriever(document_store)
    elif args.retriever == "dpr":
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=True,
            embed_title=False,
        )
    elif args.retriever == "embedding":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            model_format="sentence_transformers"
        )

    else:
        raise ValueError("Unknown retriever")

    retriever.debug = True
    if args.rebuild_dataset:
        if args.retriever != "bm25":
            document_store.update_embeddings(retriever)

    if not args.notraining and args.retriever == "dpr":
        # train DPR
        train_filename = "data/retrieval/train.json"
        test_filename = "data/retrieval/test.json"
        doc_dir = "data/retrieval/"
        save_dir = "../model_save/retrieval/" + args.retriever + "/"
        retriever.train(
            data_dir=doc_dir,
            train_filename=train_filename,
            dev_filename=test_filename,
            test_filename=test_filename,
            n_epochs=1,
            batch_size=16,
            grad_acc_steps=8,
            save_dir=save_dir,
            evaluate_every=3000,
            embed_title=True,
            num_positives=1,
            num_hard_negatives=1,
        )
        reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=None)

    if True:
        evaluate_manual(args, test_filepaths, retriever)
    else:
        evaluate_haystack(args, test_filepaths, retriever, document_store,
            doc_index, label_index, preprocessor)


def evaluate_manual(args, test_filepaths, retriever):
    questions, answers, contexts = load_labels(args, test_filepaths)
    # shuffle the order of questions and answers
    both = list(zip(questions, answers))
    random.seed(42)
    random.shuffle(both)
    questions, answers = zip(*both)
    k_list = [1, 5, 20, 50, 100, 200, 500]
    mrr = []
    for i in tqdm.tqdm(range(len(questions))):
        question = questions[i]
        retrieved_docs = retriever.retrieve(query=question, top_k=max(k_list))
        mrr.append(docs_contain_citation(retrieved_docs, answers[i]))
        if (i+1)%100 == 0:
            print_metrics(mrr, k_list)
    print_metrics(mrr, k_list)


def recall_at_k(reciprocal_ranks, k_list):
    recall_at_k = {}
    for k in k_list:
        recall_at_k[k] = len([1 for r in reciprocal_ranks if r <= k and r > -1]) / len(reciprocal_ranks)
    return recall_at_k


def evaluate_haystack(args, test_filepaths, retriever, document_store,
    doc_index, label_index, preprocessor):
    docs2answers = Docs2Answers()
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipe.add_node(component=docs2answers, name="Doc2Answers", inputs=["Retriever"])

    questions, answers, contexts = load_labels(args, test_filepaths)
    generate_squad_json(questions, answers, contexts, "evalset.json")

    document_store.add_eval_data(
        filename="evalset.json",
        # filename="data/tutorial5/nq_dev_subset_v2.json",
        doc_index=doc_index,
        label_index=label_index,
        preprocessor=preprocessor,
        open_domain=False
    )

    # run evaluation
    eval_labels = document_store.get_all_labels_aggregated()
    eval_result = pipe.eval(labels=eval_labels, params={"Retriever": {"top_k": 5}})
    print(eval_result)


if __name__ == "__main__":
    main()