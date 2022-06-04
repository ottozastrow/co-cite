# imports
import os
import pandas as pd
import tqdm
from datasets import Dataset


args = None  # put cmd arguments in here

# to simplify: use only longest string of citation variants
def get_longest(citations):
    longest_variants = []
    for citation_variants in citations:
        longest = max(citation_variants, key=len)
        longest_variants.append(longest)
    return longest_variants


def read_files_to_df(args):
    # read file
    # get list of filepaths in data_dir
    filepaths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json')]

    dfs = [] # an empty list to store the data frames

    if args.miniature_dataset:
        filepaths = filepaths[:args.miniature_dataset_size]

    print("reading json files into df")
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
    print("train samples: ", len(dataset['train']), 
          "test samples: ", len(dataset['test']))
    return dataset


def load_dataset(passedargs):
    global args
    args = passedargs
    df = read_files_to_df(args)
    df = preprocess_data(df)
    return df
    