import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np
import wandb


def plot_precision_recall(prefix, matches, scores, top_ks, buckets=40) -> None:
    try:
        # find threshold such that scores is split into equal buckets
        scores_sorted = np.sort(scores)
        scores_sorted = scores_sorted[::-1]
        # thresholds are periodically taken across sorted scores
        thresholds = [scores_sorted[(len(scores) // buckets) * i]
                    for i in range(buckets)]

        for k in top_ks:
            # for every threshold, compute the precision
            precisions = []
            for threshold in thresholds:
                # compute the number of true positives
                true_positives = np.sum(matches[k] & (scores >= threshold))
                # compute the number of false positives
                false_positives = np.sum(np.logical_not(
                    matches[k]) & (scores >= threshold))
                # compute the precision
                precision = true_positives / (true_positives + false_positives)
                # append the precision to the list of precisions
                precisions.append(precision)

            # plot curve
            # plt.legend(["top " + str(k)])
            plt.plot([i/buckets for i in range(buckets)], precisions, label="top " + str(k))
            # plot with legend
            # yaxsis, xaxis, title
        plt.xlabel('recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')

        wandb.log({prefix + "_precision_recall": plt})

    except Exception as e:
        print("WARNING: exception in plot precision recall:", e)


def plot_accs_per_occurrence(df, columns) -> None:
    df = df.sort_values(by="label_occurrences", ascending=False)
    # show average of new_segment_acc per quantile occurences
    # compute average of segment_acc over 10 buckets of label_occurrences
    # convert df to list
    df_list = df.to_dict("records")
    num_buckets = 40
    num_buckets = min(num_buckets, len(df_list))
    df_list_split = np.array_split(df_list, num_buckets)
    quantiles = {}
    x_titles = []

    for column in columns:
        quantiles[column] = []
        for i, df_list_i in enumerate(df_list_split):
            index = str(df_list_i[0]["label_occurrences"]) + " - "\
                + str(df_list_i[-1]["label_occurrences"])
            x_titles.append(index)
            # compute average of segment_acc over 10 buckets of label_occurrences
            avg = np.array(
                    [row[column] for row in df_list_i]
            ).mean()
            quantiles[column].append(avg)

    fig = go.Figure()
    for column in columns:
        fig.add_bar(
            name=column,
            x=x_titles,
            y=list(quantiles[column]))

    # fig.update_layout(barmode='stack')
    # fig.update_layout(barmode="relative")
    fig.update_layout(title_text="Average Segment Accuracy per Quantile Occurrences")

    # x axis = occurrence range
    fig.update_xaxes(title_text="Occurrence Range")
    # y axis = average segment accuracy
    fig.update_yaxes(title_text="Average Segment Accuracy")
    # fig.show()
    wandb.log({"avg_segment_acc_per_quantile": fig})
