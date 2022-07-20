
import time

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tqdm
import wandb
import pandas as pd

import metrics
import plots
import csv



def rearrange_model_generate(predictions, args):
    # annoying reformatting because of transformers.model.generate output is flattened (batchsize x beams)
    beams = [[] for i in range(args.topk)]
    assert len(predictions) == args.topk * args.batchsize,\
        "assuming this format as output from generate()" + str(len(predictions))
    # assuming predictions is batchsize x beam_num x tokens
    for i in range(args.batchsize):
        for j in range(args.topk):
            beams[j].append(predictions[i*args.topk + j])
    return beams


def tokens_2_words(tokenizer, predictions, labels, inputs=None):
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    if inputs is not None:
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    else:
        decoded_inputs = None
    return decoded_predictions, decoded_labels, decoded_inputs


def generate_batch(model, inputs, args):
    # timeit
    start = time.time()
    # generate predictions
    modeloutdict = model.generate(
        inputs, num_beams=args.topk,
        num_return_sequences=args.topk,
        do_sample=args.sample_decoding, top_k=args.topk, early_stopping=True,
        output_scores=True, return_dict_in_generate=True, max_length=args.output_tokens)
    
    # timeit
    end = time.time()
    latency = end - start

    scores = modeloutdict["scores"]
    predictions = modeloutdict["sequences"]


    if args.topk != 1:
        beams = rearrange_model_generate(predictions, args)
        scores = list(scores.numpy())
        scores = [scores[i] for i in range(len(scores)) if i%args.topk == 0]  # TODO remove or check if highest score is at index =0
    else:
        # when args.topk == 1, predictions is a tuple of logits (tensor)
        # of shape seqlen x batchsize x vocabsize

        scores = tf.convert_to_tensor(scores)
        probabilities = tf.reduce_max(scores, axis=2)
        mean_probabilites = tf.reduce_mean(probabilities, axis=0)

        scores = mean_probabilites.numpy().tolist()

        beams = [predictions]
    

    return beams, scores, latency


def evaluate(model, dataset, metric_fn, prefix, args, top_ks, tokenizer):
    print("starting eval" + prefix)
    samples_table = []

    metric_outputs = []
    segment_accs = []  # a given target may contain several segments.
    segment_accs_no_subsections = []
    # here accuracy is computed for each segment
    # if top_ks isnatnce of tuple
    if isinstance(top_ks, tuple):
        top_ks = top_ks[0]  # unexplainable bug causes this
    all_matches = {}
    for k in top_ks:
        all_matches[k] = []
    
    # load occurences dict from csv args.data_dir/occurrences.csv
    with open(args.parquet_data_dir_name + "/occurrences.csv", "r") as f:
        reader = csv.reader(f)
        occurences_map = {rows[0]: int(rows[1]) for rows in reader}
    print("loaded occurences. found ", len(occurences_map), " entries")

    decoding_latencies = []
    all_scores = []

    
    for batch in tqdm.tqdm(dataset):
        beams, scores, latency = generate_batch(model, batch["inputs"], args)
        decoding_latencies.append(latency)
        decoded_labels, beams, decoded_inputs = tokens_2_words(
            tokenizer,beams, batch["labels"], batch["inputs"])
        metric_output, matches = metric_fn(beams, decoded_labels, several_beams=True)
        all_scores += scores

        occurrences = []
        for label in decoded_labels:
            if label in occurences_map.keys():
                occurrences.append(occurences_map[label])
            else:
                print("didn't find label in occurences_map: ", label)
                occurrences.append(None)

        # add matches dict to all matches
        for k in list(matches.keys()):
            all_matches[k].extend(matches[k])

        mean_segment_accs = []

        ### segment accuracy metrics
        # iterate over elements in batches
        for i in range(len(decoded_labels)):
            segment_accs_no_subsections.extend(
                metrics.citation_segment_acc(
                    beams[0][i], decoded_labels[i],
                    remove_subsections=True, remove_subsubsections=True,
                    args=args,
                )
            )

            segment_acc = metrics.citation_segment_acc(
                beams[0][i], decoded_labels[i], args=args,
                remove_subsections=False, remove_subsubsections=True)
            segment_accs.extend(segment_acc)
            mean_segment_accs.append(np.mean(segment_acc))

        ### results table for wandb
        # change dim ordering of list of lists
        beams_reorderd = [list(i) for i in zip(*beams)]
        metric_outputs.append(metric_output)
        columns_list = [
            decoded_inputs, beams[0], decoded_labels, occurrences,
            scores, mean_segment_accs, beams_reorderd
        ]
        columns_list.extend(list(matches.values()))
        columns_list = list(columns_list)
        rows = [list(t) for t in zip(*(columns_list))]
        samples_table += rows

    columns = ["inputs", "top1 prediction", "label", "label_occurrences",
               "scores", "segment_acc", "all_topk_predictions"]
    topk_keys = ["top-" + str(i) for i in all_matches.keys()]
    columns.extend(topk_keys)
    wandb_table = wandb.Table(columns=columns, data=samples_table)
    wandb.log({prefix + "demo": wandb_table})

    # log avg decoding latency to wandb
    wandb.log({prefix + "eval_batch_decoding_latency": np.mean(decoding_latencies)})

    results = mean_over_metrics_batches(metric_outputs)
    results["segment_acc"] = np.mean(segment_accs)
    results["segment_acc_no_subsections"] = np.mean(segment_accs_no_subsections)
    print({prefix: results})
    wandb.log({prefix: results})

    all_scores = np.array(all_scores)
    try:
        plot = plots.plot_precision_recall(all_matches, all_scores, top_ks=top_ks)
        wandb.log({prefix + "_precision_recall": plot})

    except Exception as e:
        print("WARNING: exception in plot precision recall:", e)

    table = pd.DataFrame(samples_table, columns=columns)

    plots.plot_accs_per_occurrence(table, columns=["segment_acc"] + topk_keys)

    return results


def mean_over_metrics_batches(batchwise_metrics):
    # iterate through keys and items in metric_outputs and compute mean
    # create numpy array from list of dicts
    keys_list = list(batchwise_metrics[0].keys())
    metric_outputs = {}
    for key in keys_list:
        values = []
        for batch in batchwise_metrics:
            values.append(batch[key])
        metric_outputs[key] = np.mean(values)
    return metric_outputs
