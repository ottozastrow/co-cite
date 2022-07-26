import os
import pandas as pd
import json
import pdb
import numpy as np
# import plotly
import plotly.graph_objects as go


import config

def test_dynamic_k():
    filepath = "../../data/tables/test_demo_1_e85451fb7b21661f977c.table.json"
    # check if exists
    if not os.path.exists(filepath):
        raise Exception("filepath does not exist")
    args = config.cmd_arguments()

    table_dict = json.load(open(filepath))
    columns, data = table_dict["columns"], table_dict["data"]
    table = pd.DataFrame(data, columns=columns)

    print(dynamic_k(table, args))


def dynamic_k(table, args):
    # apply dynamic_k_per_sample to each row
    # drop all rows except 2
    # table = table[:2]
    buckets = 39
    scores_per_sample = table["scores_topk"]
    # flatten series of lists to list
    scores = [item for sublist in scores_per_sample for item in sublist]

    scores_sorted = np.sort(scores)
    scores_sorted = scores_sorted[::-1]
    # thresholds are periodically taken across sorted scores
    thresholds = [scores_sorted[(len(scores) // buckets) * i]
                for i in range(buckets)]
    thresholds.append(scores_sorted[-1])
    buckets += 1
    
    table["dynamic_k"] = table.apply(k_at_threshold, args=[-0.8, args], axis=1)
    # compute average over danymic k
    # visualize table
    # import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    # plot histogramm of dynamic_k
    plt.hist(table["dynamic_k"])
    plt.xlabel("dynamic_k")
    plt.ylabel("count")
    plt.title("Histogram of dynamic_k")
    plt.show()

    mean_k = table["dynamic_k"].mean()
    print("mean_k", mean_k)
    return mean_k


def k_at_threshold(row, threshold: float, args):
    """
    K = 20, k \in {0, ..., K}$   find the max k for which sum(logits[:k]) / k > threshold
    scores is a series or array or list of floats with K == args.topk
    """

    scores = row["scores_topk"]
    K = len(scores)
    # sort scores reversely
    scores = np.sort(scores)[::-1]

    y = [sum(scores[:k]) for k in range(1, K+1)]
    x = np.arange(0, K)

    # import matplotlib.pyplot as plt
    # plt.plot(x, y)
    # plt.plot(x, [(i+1)*threshold for i in x])

    found_index = -1
    for i in range(K):
        if sum(scores[:i+1]) / (i+1) < threshold:
            found_index = i
            break
    # plt.plot(found_index, y[found_index], "ro")
    # plt.show()
    return found_index

test_dynamic_k()