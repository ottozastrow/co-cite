import logging
from transformers import BertTokenizer, BertModel

# get data
import os
import pandas as pd
import matplotlib.pyplot as plt
filename = "../../external_projects/data_raw_bva/supreme_court_opinions/20220419_lexis_opinions_filtered_op_length_mini.csv"

# read csv into dataframe
df = pd.read_csv(filename)

df['citations_raw'] = df['citations_raw'].apply(lambda d: d[2:-2].split(",") if isinstance(d, str) else [])
df['num_citations'] = df['citations_raw'].apply(lambda x: len(x))

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px

#Libraries for preprocessing
from gensim.parsing.preprocessing import remove_stopwords
import string
import webcolors

#Libraries for clustering
from sklearn.cluster import KMeans

# initialize model
from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_embedding(citations):
    embeddings = []
    for citation in citations:
        inputs = tokenizer(citation, return_tensors="pt")
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state)
    return embeddings

# df['embeddings'] = df["citations_raw"].apply(get_embedding)
embeddings = []
for i in range(len(df)):
    embed = get_embedding(df['citations_raw'].loc[i])
    embeddings.append(embed)
    print(embed)

print(embeddings)
logging.info(embeddings)
