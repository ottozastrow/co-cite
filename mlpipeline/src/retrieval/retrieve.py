from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers
from haystack.nodes import DensePassageRetriever, BM25Retriever, EmbeddingRetriever

from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from haystack.utils import launch_es
from haystack.pipelines import Pipeline
from haystack.nodes import Docs2Answers

import config
import retrieval.data_utils as data_utils

import tqdm
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


def build_document_store(args, document_store, filepaths, preprocessor):

    if args.rebuild_dataset:
        document_store.delete_all_documents()

    filebatchsize=100
    batches = [filepaths[i:i+filebatchsize] for i in range(0, len(filepaths), filebatchsize)]
    
    for i in tqdm.tqdm(range(len(batches))):
        docs = data_utils.read_docs_batch_from_json(batches[i])
        if args.retriever == "bm25":
            docs = preprocessor.process(docs)
        document_store.write_documents(docs)
    
    return document_store


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


def train_dpr(retriever, args):
    #train_filename = "train/biencoder-nq-train.json"
    test_filename = "test_dpr_dataset.json"
    train_filename = "test_dpr_dataset.json"
    # WARNING: both are set to same file for debugging

    doc_dir = f"../../data/retrieval/{args.retriever}/v1/"
    save_dir = "../../model_save/retrieval/" + args.retriever + "/"

    if not isinstance(retriever, DensePassageRetriever):
        raise ValueError("DPR only works with DPR retriever")

    retriever.train(
        data_dir=doc_dir,
        train_filename=train_filename,
        dev_filename=test_filename,
        test_filename=test_filename,
        n_epochs=1,
        batch_size=args.batchsize,
        grad_acc_steps=4,
        save_dir=save_dir,
        evaluate_every=3000,
        embed_title=False,
        num_positives=10,
        num_hard_negatives=10,
    )
    return retriever


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

    train_filepaths, test_filepaths = data_utils.load_retrieval_dataset(args)

    preprocessor = data_utils.setup_preprocessor(args)
    if args.rebuild_dataset:
        document_store = build_document_store(
            args, document_store, train_filepaths, preprocessor
        )

    ######## setup retriever #########
    if args.retriever == "bm25":
        retriever = BM25Retriever(document_store)
    elif args.retriever == "dpr":
        retriever = DensePassageRetriever(
            document_store=document_store,
            # query_embedding_model = "t5-small",
            # passage_embedding_model = "t5-small",
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            # max_seq_len_query=8,
            # max_seq_len_passage=8,
            use_gpu=True,
            embed_title=False,
            batch_size=args.batchsize,

        )
    elif args.retriever == "embedding":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            model_format="sentence_transformers"
        )

    else:
        raise ValueError("Unknown retriever")

    if not args.notraining and args.retriever == "dpr":
        retriever = train_dpr(retriever, args)

    # update embeddings
    retriever.debug = True
    if args.rebuild_dataset:
        if args.retriever != "bm25":
            document_store.update_embeddings(retriever)

    evaluate_manual(args, test_filepaths, retriever)


def evaluate_manual(args, test_filepaths, retriever):
    questions, answers = data_utils.load_labels(args, test_filepaths)
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

    questions, answers = load_labels(args, test_filepaths)
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