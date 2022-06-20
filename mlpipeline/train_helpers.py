import numpy as np
import os
import tensorflow as tf
from tensorflow.nn import ctc_beam_search_decoder
import matplotlib.pyplot as plt
import tqdm

from datasets import load_metric
import wandb

from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BeamSearchScorer
)

def normalize(x):
    # first remove year:
    # "1vet.app.49(c), 55 (1990)"
    # -> 1vet.app.49(c), 55 
    if len(x) > 2:
        if x[-1] == ")" and x[-2].isnumeric():
            x = x.rsplit("(", 1)[0]
    return x.lower().replace(" ", "").replace("(", "").replace(")", "")


def split_citation_segments(inputs):
    books = ["U.S.C.A", "C.F.R."]
    books_normed = [normalize(book) for book in books]
    # split string on comma and remove empty strings
    if len(inputs) == 0:  # if input is an ampty string
        splits = [""]
    else:
        splits = [str for str in inputs.split(",") if str != ""]
    # if any book normed appears in the citation
    # if any([book in normalize(splits[0]) for book in books_normed]):
    txt = [normalize(segment) for segment in splits]
    # else:
        # txt = [inputs]
    return txt


def citation_segment_acc(predictions, labels):
    """
    converts from list of strings of this kind
    38 U.S.C.A. 5100, 5102, 5103, 5103A, 5106, 5107
    to list of list of strings such that
    [38 U.S.C.A. 5100, 38 U.S.C.A. 5102, ...]

    then computes acc accross both
    
    """
    accs = []
    for i in range(len(labels)):
        x = predictions[i]
        y = labels[i]
        x = split_citation_segments(x)
        y = split_citation_segments(y)
        # compute accuracy
        # for every el in y, check if x has el
        contained = [el in x for el in y]
        sample_acc = np.mean(contained)
        accs.append(sample_acc)
    # batch_acc = np.mean(accs)  # TODO remove
    return accs


def logits_to_topk(logits, k):
    # use beam search to get the top k predictions
    # tf requires length x batchsize x num_classes
    logits_tf = tf.reshape(logits, (logits.shape[1], logits.shape[0], logits.shape[2]))
    sequence_length = tf.ones(logits.shape[0], dtype=tf.int32) * logits.shape[1]

    beams, log_probs = ctc_beam_search_decoder(
        logits_tf, sequence_length=sequence_length, 
        beam_width=k,
        top_paths=k
    )

    beams = [tf.sparse.to_dense(b) for b in beams]

    return beams, log_probs

class CustomMetrics():
    def __init__(self, prefix, args, top_ks):
        self.prefix = prefix
        self.args = args
        self.top_ks = top_ks

    def fast_metrics(self, beams, labels, several_beams=False) -> dict:
        """
        tupledict is a tuple of (dict(list()), list())
        its counterintuitive but I'll keep it since the huggingface
        library uses it as interface
        """

        if not several_beams:
            beams = [beams]
                
        ### accuracies
        # correctness of batchsize x beam_index
        top_ks = [1, 3, 5, 20]
        max_k = max(top_ks)
        max_k = min(max_k, len(beams)) 
        top_ks = [k for k in top_ks if k <= max_k]

        match_at_k = np.zeros((len(beams), len(labels)))
        matches = {}
        results = {}
        # iterate through all beams
        for i in range(len(beams)):
            beam = beams[i]
            for j in range(len(labels)):  # iterate through batch

                x = normalize(beam[j])
                y = normalize(labels[j])
                exact_match = x == y
                match_at_k[i, j] = exact_match

        for k in top_ks:
            matches_topk = np.any(match_at_k[:k, :], axis=0)
            matches_topk = np.array(matches_topk).astype(int)

            matches[k] = matches_topk
            topk_acc = np.mean(matches_topk)
            results[self.prefix + "top{}_acc".format(k)] = topk_acc

        top1matches = list(match_at_k[0])

        return results, matches


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, model, tokenizer):
        self.save_path = save_path
        self.model = model
        self.tokenizer = tokenizer
        self.counter = 0
        self.log_interval=20000
        self.epochcounter = 0

    def on_epoch_end(self, epoch, logs=None, incrase_epoch=True):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        name = self.save_path + "epoch_" + str(epoch)
        self.model.save_pretrained(name)
        self.tokenizer.save_pretrained(name)
        print("Saved model and toknizer to {}".format(name))
        if incrase_epoch:
            self.epochcounter += 1

    def on_train_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_interval == 0:
            self.on_epoch_end(self.epochcounter, incrase_epoch=False)


def rearrange_model_generate(predictions, args):

    # annoying reformatting because of transformers.model.generate output is flattened (batchsize x beams)
    # TODO: check whether output makes sense
    beams = [[] for i in range(args.topk)]
    assert len(predictions) == args.topk * args.batchsize,\
        "assuming this format as output from generate()" + str(len(predictions))
    # assuming predictions is batchsize x beam_num x tokens
    for i in range(args.batchsize):
        for j in range(args.topk):
            beams[j].append(predictions[i*args.topk + j])
    return beams


def batch_accuracy(predictions, labels):
    """Computes sequence wise accuracy.
    Args:
        predictions: predictions of shape (batch_size, sequence_length)
        labels: labels of shape (batch_size, sequence_length)
    Returns:
        accuracy: average accuracy of the predictions"""

    return np.mean([pred == label for pred, label in list(zip(predictions, labels))])

def tokens_2_words(tokenizer, predictions, labels, batched=True):
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
   
    ### sample outputs
    return decoded_predictions, decoded_labels


def evaluate(model, dataset, metric_fn, prefix, args, top_ks, tokenizer):
    print("starting eval" + prefix)
    metric_outputs = []
    all_matches = {}
    for k in top_ks:
        all_matches[k] = []
    all_scores = []
    samples_table = []
    for batch in tqdm.tqdm(dataset):
        inputs = batch["input_ids"]
        labels = batch["labels"]
        modeloutdict = model.generate(inputs, num_beams=args.topk, num_return_sequences=args.topk,
                                    output_scores=True, return_dict_in_generate=True)
        scores = modeloutdict["scores"]
        predictions = modeloutdict["sequences"]

        decoded_predictions, decoded_labels = tokens_2_words(tokenizer, predictions, labels, batched=True)

        beams = rearrange_model_generate(decoded_predictions, args)
        beams = rearrange_model_generate(decoded_predictions, args)

        metric_output, matches = metric_fn(beams, decoded_labels, several_beams=True)

        # scores = list(log_probs[:, -1].numpy())
        # add matches dict to all matches
        for k in list(matches.keys()):
            all_matches[k].extend(matches[k])
        scores = list(scores.numpy())
        scores = [scores[i] for i in range(len(scores)) if i%args.topk == 0]  # TODO remove or check if highest score is at index =0
        all_scores += scores
        
        segment_acc = citation_segment_acc(beams[0], decoded_labels)
        metric_output[prefix + "segment_accuracy"] = np.mean(segment_acc)
        
        metric_outputs.append(metric_output)
        rows=[list(t) for t in zip(beams[0], decoded_labels, scores, segment_acc )]
        samples_table += rows
    
    wandb_table = wandb.Table(columns=["prediction", "label", "scores", "segment_acc"], data=samples_table)
    wandb.log({prefix + "demo": wandb_table})

    metric_output = mean_over_metrics_batches(metric_outputs)
    all_scores = np.array(all_scores)
    plot_precision_recall(all_matches, all_scores, prefix=prefix, top_ks=top_ks)

    results = mean_over_metrics_batches(metric_outputs)

    print({'eval_' + prefix: results})
    wandb.log({'eval_' + prefix: results})


def plot_precision_recall(matches, scores, prefix, top_ks, buckets=40):
    # find threshold such that scores is split into equal buckets
    scores_sorted = np.sort(scores)
    scores_sorted = scores_sorted[::-1]
    # thresholds are periodically taken across sorted scores
    thresholds = [scores_sorted[round(len(scores) / buckets) * i] for i in range(buckets)]

    for k in top_ks:
        # for every threshold, compute the precision
        precisions = []
        for threshold in thresholds:
            # compute the number of true positives
            true_positives = np.sum(matches[k] & (scores >= threshold))
            # compute the number of false positives
            false_positives = np.sum(np.logical_not(matches[k]) & (scores >= threshold))
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


def mean_over_metrics_batches(metric_outputs):
    # iterate through keys and items in metric_outputs and compute mean
    # create numpy array from list of dicts
    keys_list = list(metric_outputs[0].keys())
    np_metric_output = []
    for batch in metric_outputs:
        np_metric_output.append([batch[key] for key in keys_list])
    np_metric_output = np.array(np_metric_output)
    np_metric_output = np.mean(np_metric_output, axis=0)
    # translate metric_outputs back to dict
    metric_output = {key: val for key, val in zip(keys_list, np_metric_output)}
    return metric_output

