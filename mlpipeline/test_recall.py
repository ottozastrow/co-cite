import numpy as np


from keras_metric_callback import plot_precision_recall, citation_segment_acc, split_citation_segments
from cocitedata import load_dataset

from config import cmd_arguments
args = cmd_arguments()
ds = load_dataset(args)["train"]
texts, labels = ds['text'], ds['label']

def test_plot_precision_recall():
    length = 100
    preds = np.random.randint(2, size=length)
    scores = np.random.uniform(-10, 1, length)
    targets = np.random.randint(2, size=length)
    plot_precision_recall(preds, scores, targets)
    assert True

def test_metrics():

    accs = citation_segment_acc(labels, labels)
    assert accs == 1.0
    accs = citation_segment_acc(["" for i in range(len(labels))], labels)
    assert accs == 0.0

    # accs = citation_segment_acc(["" for i in range(len(labels))], labels)


    x = ["38 C.F.R. 3.303,"]
    y = ["38 C.F.R. 3.303, 3.310"]
    accs = citation_segment_acc(x, y)
    assert accs == 0.5, accs

    
test_metrics()
