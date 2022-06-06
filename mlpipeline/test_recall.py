import numpy as np

import matplotlib.pyplot as plt
length = 100
preds = np.random.randint(2, size=length)
scores = np.random.uniform(-10, 1, length)
targets = np.random.randint(2, size=length)


def plot_precision_recall(preds, scores, targets, buckets=40):
    matches = preds == targets
    # create thresholds
    min, max = np.min(scores), np.max(scores)
    thresholds = np.linspace(min, max, buckets)

    # for every threshold, compute the precision
    precisions = []
    for threshold in thresholds:
        # compute the number of true positives
        true_positives = np.sum(matches & (scores >= threshold))
        # compute the number of false positives
        false_positives = np.sum((preds != targets) & (scores >= threshold))
        # compute the precision
        precision = true_positives / (true_positives + false_positives)
        # append the precision to the list of precisions
        precisions.append(precision)

    # plot curve
    plt.plot(thresholds, precisions)
    # yaxsis, xaxis, title
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    wandb.log({"precision_recall": plt})

plot_precision_recall(preds, scores, targets)