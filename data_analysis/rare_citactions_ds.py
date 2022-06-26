# import package from mlpipeline
from config import cmd_arguments
import cocitedata
import matplotlib.pyplot as plt
import os
import pandas as pd
import tqdm

def load_data():
    testargs = ["--data_dir", "../../external_projects/bva-citation-prediction/data/preprocessed-cached/preprocessed-cached-v4/"]

    args = cmd_arguments(testargs=testargs)
    data_dir_name = cocitedata.dataset_filepath(args)
    cocitedata.generate_ds_if_not_cached(data_dir_name, args)
    # df = cocitedata.parquet_to_dataframe(data_dir_name, args)
    parquet_files = [os.path.join(data_dir_name, f) for f in os.listdir(data_dir_name) if f.endswith('.parquet')]
    return parquet_files


def count():
    files = load_data()[:10]

    counts = pd.Series(dtype=int)
    for path in tqdm.tqdm(files):
        df = pd.read_parquet(path)
        counts = counts.add(df['label'].value_counts(), fill_value=0)
    counts.astype(int)
    # sort counts
    counts = counts.sort_values(ascending=False)

    relative_counts = counts / counts.sum()
    print(counts.sum())
    # plt title: most common
    plt.title("Most common label")
    relative_counts[:20].plot(kind="bar")
    plt.show()

    plt.title("least common label")
    relative_counts[-10:].plot(kind="bar")
    plt.show()



if __name__ == "__main__":
    count()