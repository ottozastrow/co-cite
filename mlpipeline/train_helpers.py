import numpy as np
import os
import tensorflow as tf
from tensorflow.nn import ctc_beam_search_decoder
import timeit
import matplotlib.pyplot as plt

from datasets import load_metric
import wandb

from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BeamSearchScorer
)


def split_citation_segments(inputs):
    books = ["U.S.C.A", "C.F.R."]
    normalize = lambda x: x.lower().replace(" ", "")
    books_normed = [normalize(book) for book in books]
    # split string on comma and remove empty strings
    if len(inputs) == 0:  # if input is an ampty string
        splits = [""]
    else:
        splits = [str for str in inputs.split(",") if str != ""]
    # if any book normed appears in the citation
    if any([book in normalize(splits[0]) for book in books_normed]):
        txt = [segment.strip().lower() for segment in splits]
    else:
        txt = [inputs]
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
    batch_acc = np.mean(accs)
    return batch_acc


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
    def __init__(self, prefix, args):
        self.prefix = prefix
        self.args = args

    def fast_metrics(self, tupledict) -> dict:
        """
        tupledict is a tuple of (dict(list()), list())
        its counterintuitive but I'll keep it since the huggingface
        library uses it as interface
        """

        beams = tupledict[0]["sequences"]
        # beams is a list of 2d tensors k x batch_size x seq_len
        if not isinstance(beams, list):
            beams = [beams]
        labels = tupledict[1]
        # convert labels from tf int64 to int32
        labels = tf.cast(labels, tf.int32)
                
        ### accuracies
        # correctness of batchsize x beam_index
        top_ks = [1, 3, 5, 20]
        max_k = max(top_ks)
        max_k = min(max_k, len(beams)) 
        top_ks = [k for k in top_ks if k <= max_k]

        match_at_k = np.zeros((len(beams), len(labels)))
        results = {}
        for i in range(len(beams)):
            batch_accs = []
            beam = beams[i]
            for j in range(len(labels)):  # iterate through batch
                beam_length = min(
                    np.count_nonzero(beam[j]), 
                    np.count_nonzero(labels[j]))
                
                cropped_beam = beam[j][:beam_length]
                cropped_label = labels[j][:beam_length]

                matches = cropped_beam == cropped_label
                exact_match = all(matches)
                batch_accs.append(exact_match)
                match_at_k[i, j] = exact_match
            if i == 0:
                batch_acc = np.mean(batch_accs)
                results[self.prefix + 'accuracy'] = batch_acc

        for k in top_ks:
            topk_acc = np.mean(np.any(match_at_k[:k, :], axis=0))
            results[self.prefix + "top{}_acc".format(k)] = topk_acc

        return results, list(match_at_k[0])


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, model, tokenizer):
        self.save_path = save_path
        self.model = model
        self.tokenizer = tokenizer
    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        name = self.save_path + "epoch_" + str(epoch)
        self.model.save_pretrained(name)
        self.tokenizer.save_pretrained(name)
        print("Saved model and toknizer to {}".format(name))


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


def batch_accuracy(predictions, labels):
    """Computes sequence wise accuracy.
    Args:
        predictions: predictions of shape (batch_size, sequence_length)
        labels: labels of shape (batch_size, sequence_length)
    Returns:
        accuracy: average accuracy of the predictions"""

    return np.mean([pred == label for pred, label in list(zip(predictions, labels))])

def tokens_2_words(tokenizer, predictions, labels):
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    ### sample outputs
    return decoded_labels, decoded_predictions

def create_metrics_fn(prefix, tokenizer, model, args, fast_metrics_only=False):
    """Function that computes all metrics for the given dataset.
    Args:
        prefix: name prefix for the metrics (e.g. "test_")
        tokenizer: tokenizer to use for the metrics
    Returns:
        results: dictionary with all metrics
        logs to wandb
    """

    rouge_metric = load_metric("rouge")
    import timeit
    def metrics_fn(data, topk=1):
        # predictions, labels = data
        labels = data[1]
        predictions = data[0]["sequences"]
        
        results = {}
        
        if not fast_metrics_only:
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            ### sample outputs
            my_table = wandb.Table(columns=["prediction", "groundtruth"], 
            data=[list(t) for t in zip(decoded_predictions, decoded_labels)])
            wandb.log({prefix + "demo": my_table})
            
            ### Compute ROUGE
            # measure time with timeit
            # if topk == 1:
            #     result = rouge_metric.compute(predictions=decoded_predictions, references=decoded_labels)
            #     start = timeit.default_timer()
            #     end = timeit.default_timer()
            #     print("rouge time: ", end - start)
            #     results =  {(prefix + key): value.mid.fmeasure * 100 for key, value in result.items()}
            #     wandb.log({prefix + "rouge": results})
            

        ### accuracy
        # if topk != 1:
        #     accuracy_per_sample = []
        #     for i in range(topk):
        #         accuracy_per_sample.append(batch_accuracy(predictions[i], labels))
            
        #     results[prefix + 'acc_top' + str(topk)] = np.mean(np.array(accuracy_per_sample))
        #     wandb.log({prefix + 'acc_top' + str(topk): results[prefix + 'acc_top' + str(topk)]})
        
        # results[prefix + 'acc'] = batch_accuracy(predictions[0], labels)
    
        # wandb.log({prefix + "accuracy": results[prefix + 'acc']})


        return results
    return metrics_fn
