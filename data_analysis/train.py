import argparse
import os
import pandas as pd
import tqdm

import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, load_metric

# add argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--modelname", help="name on huggingface model hub", default="t5-small")
parser.add_argument("--epochs", type=int, help="number of epochs", default=20)
parser.add_argument("--batchsize", type=int, help="batch size", default=16)
parser.add_argument("--contextlength", type=int, help="context length", default=300)
parser.add_argument("--miniaturedataset", 
                    help="for debugging only use 20 samples", action="store_true")

args = parser.parse_args()


# to simplify: use only longest string of citation variants
def get_longest(citations):
    longest_variants = []
    for citation_variants in citations:
        longest = max(citation_variants, key=len)
        longest_variants.append(longest)
    return longest_variants


def load_ds(data_dir):
    # read file
    # get list of filepaths in data_dir
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]

    dfs = [] # an empty list to store the data frames

    if args.miniaturedataset:
        filepaths = filepaths[:20]
    
    for file in tqdm.tqdm(filepaths):
        data = pd.read_json(file, lines=True) # read data frame from json file
        dfs.append(data) # append the data frame to the list

    df = pd.concat(dfs, ignore_index=True)
    df['longest_citations'] = df['citation_vocab'].apply(get_longest)
    # longest citations is a list of citations instead of a list of a list of citations
    return df


def find_all_indexes(txt, substr):
    # find indexes of all substrings substr in txt
    indexes = []
    i = -1
    while True:
        i = txt.find(substr, i + 1)
        if i == -1:
            break
        indexes.append(i)
    return indexes
    

import pdb
def get_citation_context(x):
    """Extract inputs and targets from a dataframe row"""
    inputs = []
    targets = []
    contextlength = args.contextlength
    indexes = find_all_indexes(x['txt'], "@cit@")  # TODO replace compuationally iniefficient method
    
    for i in range(len(x['citation_vocab'])):
        # variant_index, _ = max(enumerate(x['citation_vocab'][i]), key=lambda x: len(x[1]))
        # variant_index = 0  # TODO understand why there are several indices and replace this line
        # index = x['citation_indices'][i][variant_index]
        index = indexes[i]
        start_index = max(0, index-contextlength)
        stop_index = index  # TODO: also look at text after citation
        context = x['txt'][start_index:stop_index]
        citation = x['citation_texts'][i]
        inputs.append(context)
        targets.append(citation)

    x['inputs'] = inputs
    x['targets'] = targets
    return x

def preprocess_data(df):
    # read a fixed length context around each citation
    df = df.apply(get_citation_context, axis=1)

    # turn series of lists into series
    inputs_series = df['inputs'].apply(pd.Series).stack().reset_index(drop = True)
    targets_series = df['targets'].apply(pd.Series).stack().reset_index(drop = True)
    columns = {'text' : inputs_series,
               'label' : targets_series}
    df_pairs = pd.DataFrame(columns)
    dataset = Dataset.from_pandas(df_pairs)
    dataset = dataset.train_test_split(test_size=0.2)
    print(len(dataset['train']), len(dataset['test']))
    return dataset


def tokenize_function(examples):
    inputs = [input for input in examples['text']]
    targets = [target for target in examples['label']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)


    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=32, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    data_dir = "../../external_projects/bva-citation-prediction/data/preprocessed-cached/preprocessed-cached-v4-mini/"

    df = load_ds(data_dir)
    dataset = preprocess_data(df)

    # initialize model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.modelname)
    tokenizer = AutoTokenizer.from_pretrained(args.modelname)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)


    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batchsize,
        per_device_eval_batch_size=args.batchsize,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        fp16=False,
    )
    # predictions = trainer.predict(tokenized_datasets["test"])
    metric_bleu = load_metric("bleu")
    metric_acc = load_metric("accuracy")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits[0], axis=-1)
        results = metric_bleu.compute(predictions=[predictions], references=[labels])

        # sequence level accuracy
        # first element wise comparison, if there is a single false value, then the whole sequence is wrong
        sample_wise_acc = np.equal(predictions, labels).all(axis=1)
        results["accuracy"] = np.mean(sample_wise_acc)
        
        # sample outputs
        # sample_outputs = tokenizer.batch_decode(
        #     predictions, 
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=True)
        # # outputs = [output.replace("@cit@", "") for output in outputs]
        # sample_labels = tokenizer.batch_decode(
        #     labels, 
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=True)

        # results["samples"] = list(zip(sample_outputs, sample_labels))
        
        return results

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


    

