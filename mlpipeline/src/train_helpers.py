import numpy as np
import os
from sympy import Q
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM
from tensorflow.nn import ctc_beam_search_decoder
import matplotlib.pyplot as plt
import tqdm
import re
import time
import wandb


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, tokenizer, args, len_train_dataset, training_step=0):
        self.model = model
        self.tokenizer = tokenizer
        self.training_step = training_step
        self.epochcounter = 0
        self.args = args

        if args.debug:
            self.log_interval = (len_train_dataset / args.batchsize) // 2
            self.log_interval = max(1, self.log_interval)
        else:
            # schedule logging 5 times per epoch
            self.log_interval = (len_train_dataset / args.batchsize) // 5

        self.save_path = "../model_save/" + args.modelname + "_" + str(wandb.run.id) + "/"
        self.save_path += "debug/" if args.debug else ""
        self.training_step_from_filepath()

    def training_step_from_filepath(self):
        self.training_step = 0
        # when continuning training from a checkpoint set to non zero.
        if "model_save" in self.args.modelname:
            # load local checkpoint instead of huggingface hub model
            # set number of train steps if possible
            if "_step_" in self.args.modelname:
                # find first int in string after substring "_steps_"
                steps_text = self.args.modelname.split("_step_")[-1]
                self.training_step = int(re.search(r'\d+', steps_text).group())


    def on_epoch_end(self, epoch, logs=None):
        self.save_model(epoch)
        self.epochcounter += 1

    def on_train_batch_end(self, batch, logs=None):
        if self.training_step % self.log_interval == 0 and self.prefix != "train_":
            self.save_model(self.epochcounter)

        self.training_step += 1

    def save_model(self, epoch):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        name = self.save_path + "epoch_" + str(epoch) + "_step_" + str(self.training_step * self.args.batchsize)
        self.model.save_pretrained(name)
        self.tokenizer.save_pretrained(name)
        print("Saved model and toknizer to {}".format(name))


def normalize(x):
    # first remove year:
    # "1vet.app.49(c), 55 (1990)"
    # -> 1vet.app.49(c), 55
    if len(x) > 2:
        if x[-1] == ")" and x[-2].isnumeric():
            x = x.rsplit("(", 1)[0]
    return x.lower().replace(" ", "").replace("(", "").replace(")", "")


def book_from_statute(statute) -> str:
    """
    returns the book from a statute
    38 U.S.C.A. §§ 5100, 5102, 5103,  -> 38u.s.c.a.
    if only u.s.c. is given, return u.s.c.a.
    # some statutes are missing their §. e.g. 38 C.F.R. 3.321(b)(1)
    """
    statute = statute.replace(" ", "").lower()

    if len(statute.split("§")) == 1:
        statute = statute.replace("38u.s.c.a.", "38u.s.c.a.§")
        statute = statute.replace("38cfr", "38cfr§")
        statute = statute.replace("38c.f.r.", "38c.f.r.§")

    book = statute.split("§")[0]
    if book == "38u.s.c.":
        book = "38u.s.c.a."
    book_no_dot = book.replace(".", "")
    if "cfr" in book_no_dot:
        book = "38 C.F.R."
    elif "usca" in book_no_dot:
        book = "38 U.S.C.A."
    return book


def remove_last_numeric_brackets(x):
    """
    removes last bracket from inputs, if the content is numeric
    """
    if len(x) > 2 and x[-1] == ")" and x[-2].isnumeric():
        x = x.rsplit("(", 1)[0]
    return x


def sections_from_statute(
        statute,
        remove_subsubsections,
        remove_subsections=True
        ) -> list[str]:
    """
    returns the book from a statute
    38 U.S.C.A. §§ 5100, 5102, 5103A,  -> [5100, 5102, 5103A]
    """
    statute = statute.replace(" ", "").lower()

    if len(statute.split("§")) == 1:
        statute = statute.replace("38u.s.c.a.", "38u.s.c.a.§")
        statute = statute.replace("38cfr", "38cfr§")
        statute = statute.replace("38c.f.r.", "38c.f.r.§")

    sections = statute.split("§")[-1]
    sections = sections.split(",")
    if remove_subsubsections:
        # remove trailing brackets from sections if the content is numeric
        sections = [remove_last_numeric_brackets(section)
                    for section in sections]

    # remove leftover brackets (but leave content)
    sections = [section.replace("(", "").replace(")", "")
                for section in sections
                if section != ""]

    if remove_subsections:
        new_sections = []
        for section in sections:
            # if last letter not numeric
            if section[-1] not in "0123456789":
                section = section[:-1]
            new_sections.append(section)
        sections = new_sections

    assert len(sections) > 0, "no sections found in statute"
    for section in sections:
        assert(len(section)) > 0, "section is empty"
    return sections


def normalize_section(section):
    return section.replace(" ", "").replace("(", "").replace(")", "").lower()


def normalize_case(inputs_orig):
    # remove trailing brackets from sections if the content is numeric
    # e.g. (1998)
    inputs = remove_last_numeric_brackets(inputs_orig)

    # check if inputs begins with see
    lowered_inputs = inputs.lower()

    law_categories = [
        ("Vet. App.", r"vet\.? ?app\.?"),
        ("F.3d", r"f\.? ?3d?"),
        ("F.2d", r"f\.? ?2d?"),
        ("F.d", r"f\.? ?d"),
        ("F. supp", r"f\.? ?supp.?"),
        ("Fed. Cir.", r"fed\.? ?cir\.?"),
    ]
    patterns = []
    for category, regex in law_categories:
        patterns.append((category, re.compile(regex)))

    """
    See Combee v. Brown, 34 F.3d 1039, 1042
    See Combee v. Brown, 34 F. 3
    normalized [combeevbrown34 f3]
    there are many such examples, where F.3d and F.3 are used interchangebly
    """
    # TODO support multiple categories per citation (always use the first one)

    for category, pattern in patterns:
        span = pattern.search(lowered_inputs)
        if span:
            participants = inputs[:span.start()].strip()

            details = inputs[span.end():].strip()
            details = details.replace(" ", "")
            details = details.lower()

            details = details.split("-")[0]
            details = details.split("(")[0]
            details = details.split(",")[0]

            # remove trailing symbols
            details = details.replace("(", "")
            details = details.replace(")", "")
            details = details.replace("-", "")
            details = details.replace(".", "")
            details = details.replace(",", "")

            inputs = participants + " " + category + " " + details
            break

    return inputs


def split_citation_segments(inputs, remove_subsubsections=True) -> list[str]:
    """
    splits a citation into segments
    38 U.S.C.A. §§ 5100A, 5102(a)(1) becomes
    [38 U.S.C.A. § 5100a, 38 U.S.C.A. § 5102a]

    or if its not a statute

    transforms from
    See Moreau v. Brown, 9 Vet. App. 389, 396 (1996)
    to
    [Moreau v. Brown, 9 Vet. App. 389, 396]
    explanations:
    - "see" or trailing commas are not part
    of the citation but often in the data
    - year numbers (1997) are not always present
    - in Vet. App. 389, 396  we have the appendix
    389 but page number 396. the page number is often not present
    """

    # regex pattern for if "see" or "eg" "in" is at beginning of string
    # has to work for "See, " "e.g. ", "See in ", ...
    useless_prefix = re.compile(r"((see|e\.g\.|in)\.?\,? )+")

    useless_prefix_span = useless_prefix.search(inputs.lower())
    if useless_prefix_span:
        if useless_prefix_span.start() == 0:
            inputs = inputs[useless_prefix_span.end():]
    try:
        x = inputs.lower()
        is_case = "§" not in x\
            and "c.f.r" not in x\
            and "u.s.c.a" not in x\
            and "u.s.c." not in x
        if not is_case:
            book = book_from_statute(inputs)
            sections = sections_from_statute(inputs, remove_subsubsections)
            segments = [book + " § " + section for section in sections]
            return segments
        else:
            inputs = normalize_case(inputs)
            return [inputs]
    except Exception as e:
        print(e)
        print("inputs:", inputs)
        return [inputs]


def citation_segment_acc(predictions, labels, args=None):
    """
    assumes batched inputs

    converts from list of strings of this kind
    38 U.S.C.A. §§ 5100, 5102, 5103, 5103A, 5106, 5107
    to list of list of strings such that
    [38 U.S.C.A. § 5100, 38 U.S.C.A. § 5102, ...]

    then computes acc accross both

    """
    accs = []
    for i in range(len(labels)):
        x = predictions[i]
        y = labels[i]
        if args:
            if args.diffsearchindex_training:
                x = x.split("[SEP]")[0]
                y = y.split("[SEP]")[0]
        x = split_citation_segments(x)
        y = split_citation_segments(y)
        # compute accuracy
        # builds on assumption that x and y don't contain duplicates.
        # in the BVA corpus, this is true.
        # for every el in y, check if x has el
        contained = [el in x for el in y]
        contained = []
        for yi in y:
            contains = []
            for xi in x:
                contains.append(yi in xi)
            contained.append(any(contains))
        # import pdb
        # pdb.set_trace()
        accs.extend(contained)
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


def batch_accuracy(predictions, labels):
    """Computes sequence wise accuracy.
    Args:
        predictions: predictions of shape (batch_size, sequence_length)
        labels: labels of shape (batch_size, sequence_length)
    Returns:
        accuracy: average accuracy of the predictions"""

    return np.mean([pred == label for pred, label in list(zip(predictions, labels))])

def tokens_2_words(tokenizer, predictions, labels, inputs=None):
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    if inputs is not None:
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    else:
        decoded_inputs = None
    return decoded_predictions, decoded_labels, decoded_inputs


def evaluate(model, dataset, metric_fn, prefix, args, top_ks, tokenizer):
    print("starting eval" + prefix)
    metric_outputs = []
    all_matches = {}
    segment_accs = []  # a given target may contain several segments. here accuracy is computed for each segment
    # if top_ks isnatnce of tuple
    if isinstance(top_ks, tuple):
        top_ks = top_ks[0] # unexplainable bug causes this
    for k in top_ks:
        all_matches[k] = []
    
    decoding_latencies = []
    all_scores = []
    samples_table = []
    for batch in tqdm.tqdm(dataset):
        inputs = batch["input_ids"]
        labels = batch["labels"]

        # timeit
        start = time.time()
        # generate predictions
        if not args.fast_predict:
            modeloutdict = model.generate(
                inputs, num_beams=args.topk,
                num_return_sequences=args.topk,
                do_sample=args.sample_decoding, top_k=args.topk,
                output_scores=True, return_dict_in_generate=True, max_length=args.output_tokens)
        else:
            assert not (args.topk > 1 or args.sample_decoding), "fast predict doesn't support beam search"
            
            modeloutput = model.predict({"input_ids": inputs, "decoder_input_ids": inputs})
            logits = modeloutput.logits

            sequences = tf.argmax(logits, axis=-1)
            modeloutdict = {
                "sequences": sequences,
                "scores": modeloutput.logits
            }
        # timeit
        end = time.time()
        decoding_latencies.append(end - start)

        scores = modeloutdict["scores"]
        predictions = modeloutdict["sequences"]

        decoded_predictions, decoded_labels, decoded_inputs = tokens_2_words(tokenizer, predictions, labels, inputs=inputs)

        if args.topk != 1:
            beams = rearrange_model_generate(decoded_predictions, args)
            scores = list(scores.numpy())
            scores = [scores[i] for i in range(len(scores)) if i%args.topk == 0]  # TODO remove or check if highest score is at index =0
        else:
            # when args.topk == 1, predictions is a tuple of logits (tensor)
            # of shape seqlen x batchsize x vocabsize

            scores = tf.convert_to_tensor(scores)
            probabilities = tf.reduce_max(scores, axis=2)
            mean_probabilites = tf.reduce_mean(probabilities, axis=0)

            scores = mean_probabilites.numpy().tolist()

            beams = [decoded_predictions]

        metric_output, matches = metric_fn(beams, decoded_labels, several_beams=True)

        # add matches dict to all matches
        for k in list(matches.keys()):
            all_matches[k].extend(matches[k])
        
        all_scores += scores
        segment_acc = citation_segment_acc(beams[0], decoded_labels, args)
        segment_accs.append(segment_acc)

        # change dim ordering of list of lists
        beams_reorderd = [list(i) for i in zip(*beams)]
        
        metric_outputs.append(metric_output)
        rows = [list(t) for t in zip(
                    decoded_inputs, beams[0], decoded_labels,
                    scores, segment_acc, beams_reorderd
                )]
        samples_table += rows
    columns = ["inputs", "top1 prediction", "label",
               "scores", "segment_acc", "all_topk_predictions"]
    wandb_table = wandb.Table(columns=columns, data=samples_table)
    wandb.log({prefix + "demo": wandb_table})

    # log avg decoding latency to wandb
    wandb.log({prefix + "eval_batch_decoding_latency": np.mean(decoding_latencies)})

    results = mean_over_metrics_batches(metric_outputs)
    results["segment_acc"] = np.mean(segment_accs)
    print({prefix: results})
    wandb.log({prefix: results})

    all_scores = np.array(all_scores)
    try:
        plot = plot_precision_recall(all_matches, all_scores, top_ks=top_ks)
        wandb.log({prefix + "_precision_recall": plot})

    except:
        print("WARNING: exception in plot precision recall")
    return results


def plot_precision_recall(matches, scores, top_ks, buckets=40):
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

    return plt

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

