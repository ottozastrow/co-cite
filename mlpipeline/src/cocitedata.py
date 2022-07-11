# imports
import os
import pandas as pd
import tqdm
import datasets
from transformers import AutoTokenizer


def preprocess_json_and_save_to_parquet(data_dir_name, args):
    """json data is the format provided by the BVA dataset.
    This function converts it to parquet format.
    
    beforer converting to parquet the data is processed
    we convert from one row per document to one row per citation."""

    filepaths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json')]
    if args.samples != -1:
        filepaths = filepaths[:args.samples]

    print("reading json files into df")
    # create batches of files
    filebatchsize=100
    batches = [filepaths[i:i+filebatchsize] for i in range(0, len(filepaths), filebatchsize)]
    tmp_dir_name = data_dir_name[:-1] + "_unfinished/"
    # delete tmp dir if it exists
    if os.path.exists(tmp_dir_name):
        os.system("rm -r " + tmp_dir_name)
    os.makedirs(tmp_dir_name)
    for i in tqdm.tqdm(range(len(batches))):
        df = None
        batch = batches[i]
        for file in batch:
            data = pd.read_json(file, lines=True) # read data frame from json file
            data = preprocess_data(data, args)  # expensive operation
            if df is None:
                df = data
            else:
                df = pd.concat([df, data], copy=False, ignore_index=True)
        batch_fp = tmp_dir_name + "batch_" + str(i) + ".parquet"
        print("finished converting batch nr. ", str(i), "to df")
        df = df.to_parquet(batch_fp, compression=None)
    # rename folder from tmp_dir_name to data_dir_name
    # delete data_dir_name if it exists
    if os.path.exists(data_dir_name):
        os.system("rm -r " + data_dir_name)
    
    os.rename(tmp_dir_name, data_dir_name)
    print("saved df to parquet", data_dir_name)

def parquet_to_dataframe(parquet_dir, args):
    parquet_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    df = pd.read_parquet(parquet_files, engine='pyarrow')

    return df

def parquet_to_dataset(parquet_dir, args):
    parquet_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    df = datasets.load_dataset("parquet", data_files=parquet_files, cache_dir="../huggingface_cache/datasets/")
    df = df['train']  # load_datasets makes this necessary
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
    

def get_citation_context(x, args):
    """Extract inputs and targets from a dataframe row"""
    inputs = []
    targets = []
    contextlength = args.contextlength
    indexes = find_all_indexes(x['txt'], "@cit@")  # TODO replace compuationally iniefficient method
    
    for i in range(len(x['citation_vocab'])):
        # variant_index, _ = max(enumerate(x['citation_vocab'][i]), key=lambda x: len(x[1]))
        # variant_index = 0  # TODO understand why there are several indices and replace this line
        # index = x['citation_indices'][i][variant_index]
        try:
            index = indexes[i]
            start_index = max(0, index-contextlength)
            stop_index = index  # TODO: also look at text after citation
            context = x['txt'][start_index:stop_index]
            citation = x['citation_texts'][i]
            inputs.append(context)
            targets.append(citation)
        except:
            print("skip citation", i, indexes)

    x['inputs'] = inputs
    x['targets'] = targets
    return x

def preprocess_data(df, args):
    # read a fixed length context around each citation
    df = df.apply(get_citation_context, axis=1, args=(args,))

    # turn series of lists into series
    inputs_series = df['inputs'].apply(pd.Series).stack().reset_index(drop = True)
    targets_series = df['targets'].apply(pd.Series).stack().reset_index(drop = True)
    columns = {'text' : inputs_series,
               'label' : targets_series}
    df_pairs = pd.DataFrame(columns)
    
    return df_pairs


def create_tokenize_function(tokenizer, args):
    """Mapping function that tokanizes all relevant dataset entries."""
    def tokenize_function(examples):
            inputs = [input for input in examples['text']]
            targets = [target for target in examples['label']]
            model_inputs = tokenizer(inputs, max_length=args.input_tokens, truncation=True, padding="max_length")

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=args.output_tokens, truncation=True, padding="max_length")

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
    return tokenize_function

def generate_ds_if_not_cached(data_dir_name, args):
    # create parquet files from raw data if not already done
    if not os.path.exists(data_dir_name) or args.rebuild_dataset:
        preprocess_json_and_save_to_parquet(data_dir_name, args)
    else:
        print("parquet file already exists, loading parquet...")


def dataset_filepath(args, model_name_or_path):
    # determine name of dataset based on args
    length_str = "all" 
    if args.samples == -1:
        length_str = str(args.samples)
    assert args.data_dir[-1] == "/", "data_dir must end with '/'"
    data_dir_name = args.data_dir[:-1] + "_modelname_" + model_name_or_path +\
        "_data_len_" + length_str + "/"
    return data_dir_name


def load_dataset(args, model_name_or_path="unspecified"):
    data_dir_name = dataset_filepath(args, model_name_or_path)
    tokenized_data_dir_name = data_dir_name[:-1] + "_tokenized/"

    import wandb
    wandb.config.update({"data_dir_name": data_dir_name})
    wandb.config.update({"tokenized_data_dir_name": tokenized_data_dir_name})

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # if tokenized dataset exists load it
    if os.path.exists(tokenized_data_dir_name) and not args.rebuild_dataset:
        print("loading tokenized dataset from", tokenized_data_dir_name)
        tokenized_datasets = datasets.load_from_disk(tokenized_data_dir_name)
        print("finished loading precomputed tokenized ds")

    else:
        print("rebuilding dataset at ", tokenized_data_dir_name)
        generate_ds_if_not_cached(data_dir_name, args)
        df = parquet_to_dataset(data_dir_name, args)

        df = df.train_test_split(test_size=0.1)

        tokenized_datasets = df.map(
            create_tokenize_function(tokenizer, args=args),
            batched=True
        )
        tokenized_datasets.save_to_disk(tokenized_data_dir_name)
    
    return tokenized_datasets, tokenizer
