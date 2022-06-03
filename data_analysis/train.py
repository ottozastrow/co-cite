import os
import pandas as pd
import tqdm

from transformers import DistilBertTokenizer, DistilBertModel
from datasets import Dataset, Features


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
    for file in tqdm.tqdm(filepaths[:20]):
        data = pd.read_json(file, lines=True) # read data frame from json file
        dfs.append(data) # append the data frame to the list

    df = pd.concat(dfs, ignore_index=True)
    df['longest_citations'] = df['citation_vocab'].apply(get_longest)
    # longest citations is a list of citations instead of a list of a list of citations
    return df

import pdb
def get_citation_context(x):
    """Extract inputs and targets from a dataframe row"""
    inputs = []
    targets = []
    contextlength = 200
    for i in range(len(x['citation_vocab'])):
        # index of longest string in string list
        # variant_index, _ = max(enumerate(x['citation_vocab'][i]), key=lambda x: len(x[1]))
        variant_index = 0  # TODO understand why there are several indices and replace this line
        index = x['citation_indices'][i][variant_index]
        start_index = max(0, index-contextlength)
        stop_index = index  # TODO: also look at text after citation
        context = x['txt'][start_index:stop_index]
        citation = x['citation_vocab'][i][variant_index]
        inputs.append(context)
        targets.append(citation)

    x['inputs'] = pd.Series(inputs)
    x['targets'] = pd.Series(targets)
    pdb.set_trace()
    return x


if __name__ == "__main__":
    # print(get_citation_context(df.loc[0]))
    data_dir = "../../external_projects/bva-citation-prediction/data/preprocessed-cached/preprocessed-cached-v4-mini/"

    df = load_ds(data_dir)
    df = df.apply(get_citation_context, axis=1)
    columns = {'inputs' : df['inputs'].values.flatten(), 
               'targets' : df['targets'].values.flatten()}
    df_pairs = pd.DataFrame(columns)
    # df_pairs['context'] = df_pairs[0][].apply(lambda x: x.lower())

    print(df.head())
    import pdb
    pdb.set_trace()
    train_data = Dataset.from_pandas(df, split='train[:70%]')
    val_data = Dataset.from_pandas(df, split='train[70%:85]')
    test_data = Dataset.from_pandas(df, split='train[85%:]')


    # # initialize model
    # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    # model = DistilBertModel.from_pretrained("distilbert-base-uncased")
