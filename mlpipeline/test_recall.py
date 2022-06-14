import numpy as np
from transformers import DataCollatorForSeq2Seq, TFAutoModelForSeq2SeqLM


from keras_metric_callback import plot_precision_recall
from train_helpers import CustomMetrics, citation_segment_acc
from cocitedata import load_dataset

from config import cmd_arguments
args = cmd_arguments(debug=False)
args.rebuild_dataset=False

tokenized_datasets, ds, tokenizer = load_dataset(args)
ds = ds["train"]
texts, labels = ds['text'], ds['label']
model = TFAutoModelForSeq2SeqLM.from_pretrained(args.modelname)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")
generation_train_dataset = (
    tokenized_datasets["train"]
    .select(list(range(2)))
    .to_tf_dataset(
        batch_size=args.batchsize,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        collate_fn=data_collator,
    )
)


def test_plot_precision_recall():
    length = 100
    preds = np.random.randint(2, size=length)
    scores = np.random.uniform(-10, 1, length)
    targets = np.random.randint(2, size=length)
    plot_precision_recall(preds, scores, targets)
    assert True

def test_segment_metric():

    accs = citation_segment_acc(labels, labels)
    assert accs == 1.0
    accs = citation_segment_acc(["" for i in range(len(labels))], labels)
    assert accs == 0.0, accs

    # accs = citation_segment_acc(["" for i in range(len(labels))], labels)

    x = ["38 C.F.R. 3.303,"]
    y = ["38 C.F.R. 3.303, 3.310"]
    accs = citation_segment_acc(x, y)
    assert accs == 0.5, accs


def test_acc_metric():
    for batch in generation_train_dataset:
        inputs = batch["input_ids"]
        labels = batch["labels"]


        fn = CustomMetrics(prefix="test_", args=args).fast_metrics
        to_tupledict = lambda x, y: ({"sequences":x}, y)
        
        metrics, matches_at_k = fn(to_tupledict(labels, labels))
        for key in metrics.keys():
            assert metrics[key] == 1.0

        # empty predictions should have 0.0 acc
        metrics, matches_at_k = fn(to_tupledict(inputs, labels))
        for key in metrics.keys():
            assert metrics[key] == 0.0

        break

    
test_segment_metric()
test_acc_metric()
