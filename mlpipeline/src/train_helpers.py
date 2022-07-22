
import time
import collections

import numpy as np
import tensorflow as tf
import tqdm
import wandb
import pandas as pd

import metrics
import plots
import csv


def rearrange_model_generate(predictions, args) -> list:
    """
    annoying reformatting because of transformers.model.generate output is flattened (batchsize x beams)
    """
    if args.topk != 1:
        beams: list = [[] for i in range(args.topk)]
        for i in range(args.topk):
            beams[i] = predictions[i::args.topk]
        return beams
    else:
        return [predictions]


def tokens_2_words(tokenizer, predictions, labels, inputs=None):
    """Use tokenizer to tokenize predictions, labels and inputs."""
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    if inputs is not None:
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    else:
        decoded_inputs = None
    return decoded_predictions, decoded_labels, decoded_inputs


def generate_batch(model, inputs, args):
    """Runs model.generate on a batch of inputs."""
    # timeit
    start = time.time()
    # generate predictions
    modeloutdict = model.generate(
        inputs, num_beams=args.topk, use_cache=True,
        num_return_sequences=args.topk,
        do_sample=args.sample_decoding, top_k=args.topk, early_stopping=True,
        output_scores=True, return_dict_in_generate=True, max_length=args.output_tokens)
    
    # timeit
    end = time.time()
    latency = end - start

    scores = modeloutdict["scores"]
    predictions = modeloutdict["sequences"]

    if args.topk != 1:
        scores = list(scores.numpy())
        scores = [scores[i] for i in range(len(scores)) if i%args.topk == 0]  # TODO remove or check if highest score is at index =0

    else:
        # when args.topk == 1, predictions is a tuple of logits (tensor)
        # of shape seqlen x batchsize x vocabsize

        scores = tf.convert_to_tensor(scores)
        probabilities = tf.reduce_max(scores, axis=2)
        mean_probabilites = tf.reduce_mean(probabilities, axis=0)

        scores = mean_probabilites.numpy().tolist()

        predictions = [predictions]

    return predictions, scores, latency


def load_occurrences(args):
    """The citation occurrence count is computed while the dataset is generated.
    Load it as preparation for match_occurrences here."""
    # load occurences dict from csv args.data_dir/occurrences.csv
    with open(args.parquet_data_dir_name + "/occurrences.csv", "r") as f:
        reader = csv.reader(f)
        occurences_map = {rows[0]: int(rows[1]) for rows in reader}
    print("loaded occurences. found ", len(occurences_map), " entries")
    return occurences_map


def match_occurrences(decoded_labels, occurrence_map):
    """For a given label, find the number of occurrences in the dataset."""
    occurrences = []
    for label in decoded_labels:
        if label in occurrence_map.keys():
            occurrences.append(occurrence_map[label])
        else:
            print("didn't find label in occurences_map: ", label)
            occurrences.append(None)
    return occurrences


def build_wandb_table_row(
        beams, decoded_labels, decoded_inputs, scores,
        occurrences, mean_segment_accs, matches):
    ### results table for wandb
    # change dim ordering of list of lists
    beams_reorderd = [list(i) for i in zip(*beams)]
    columns_list = [
        decoded_inputs, beams[0], decoded_labels, occurrences,
        scores, mean_segment_accs, beams_reorderd
    ]
    columns_list.extend(list(matches.values()))
    columns_list = list(columns_list)
    row = [list(t) for t in zip(*(columns_list))]
    return row


def evaluate(model, dataset, prefix, args, top_ks, tokenizer):
    """Calls all metrics for this model and dataset.
    Results are logged in wandb.
    """

    print("starting eval" + prefix)
    
    samples_table = []

    # if top_ks isnatnce of tuple
    if isinstance(top_ks, tuple):
        top_ks = top_ks[0]  # unexplainable bug causes this
    all_matches = {}
    for k in top_ks:
        all_matches[k] = []
    
    occurences_map = load_occurrences(args)

    all_scores = []
    metric_outputs = collections.defaultdict(list)
    
    for batch in tqdm.tqdm(dataset):
        predictions, scores, latency = generate_batch(model, batch["input_ids"], args)
        metric_outputs["latency"].append(latency / args.eval_batchsize)
        all_scores += scores

        decoded_predictions, decoded_labels, decoded_inputs = tokens_2_words(
            tokenizer, predictions, batch["labels"], batch["input_ids"])

        beams = rearrange_model_generate(decoded_predictions, args)

        metric_output, matches = metrics.matches_at_k(beams, decoded_labels, top_ks=top_ks, several_beams=True)
        # add matches dict to all matches
        for k in list(matches.keys()):
            all_matches[k].extend(matches[k])
        
        occurrences = match_occurrences(decoded_labels, occurences_map)

        ### segment accuracy metrics
        mean_segment_accs = []
        # iterate over elements in batches
        for i in range(len(decoded_labels)):
            metric_outputs["segment_accs_no_subsections"].extend(
                metrics.citation_segment_acc(
                    beams[0][i], decoded_labels[i],
                    remove_subsections=True, remove_subsubsections=True,
                    args=args,
                )
            )

            segment_acc = metrics.citation_segment_acc(
                beams[0][i], decoded_labels[i], args=args,
                remove_subsections=False, remove_subsubsections=True)
            metric_outputs["segment_accs"].extend(segment_acc)
            
            mean_segment_accs.append(np.mean(segment_acc))
            metric_outputs["mean_segment_accs"].append(np.mean(segment_acc))

        for metric in metric_output.keys():
            metric_outputs[metric].append(metric_output[metric])
             
        samples_table += build_wandb_table_row(
            beams, decoded_labels, decoded_inputs, occurrences,
            scores, mean_segment_accs, matches)

    columns = ["inputs", "top1 prediction", "label", "label_occurrences",
               "scores", "segment_acc", "all_topk_predictions"]
    topk_keys = ["top-" + str(i) for i in all_matches.keys()]
    columns.extend(topk_keys)
    wandb_table = wandb.Table(columns=columns, data=samples_table)
    wandb.log({prefix + "demo": wandb_table})

    mean_metric_outputs = {}
    for (metric_name, metric_data) in metric_outputs.items():
        mean_metric_outputs[metric_name] = np.mean(metric_data)

    print({prefix: mean_metric_outputs})
    wandb.log({prefix: mean_metric_outputs})

    all_scores = np.array(all_scores)
    plots.plot_precision_recall(prefix, all_matches, all_scores, top_ks=top_ks)
    table = pd.DataFrame(samples_table, columns=columns)
    plots.plot_accs_per_occurrence(prefix, table, columns=["segment_acc"] + topk_keys)

    return mean_metric_outputs
