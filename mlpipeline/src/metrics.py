import numpy as np
import citation_normalization


def citation_segment_acc(
        prediction, label,
        remove_subsections, remove_subsubsections, args=None) -> list[bool]:
    """
    assumes unbatched inputs

    converts from list of strings of this kind
    38 U.S.C.A. §§ 5100, 5102, 5103, 5103A, 5106, 5107
    to list of list of strings such that
    [38 U.S.C.A. § 5100, 38 U.S.C.A. § 5102, ...]

    then computes acc accross both

    """
    accs = []
    x = prediction
    y = label
    if args:
        if args.diffsearchindex_training:
            x = x.split("[SEP]")[0]
            y = y.split("[SEP]")[0]
    x = citation_normalization.normalize_citation(
        x,
        remove_subsections=remove_subsections,
        remove_subsubsections=remove_subsubsections)
    y = citation_normalization.segmentize_citation(y)
    
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
    accs.extend(contained)
    return accs


def batch_accuracy(predictions, labels):
    """Computes sequence wise accuracy.
    Args:
        predictions: predictions of shape (batch_size, sequence_length)
        labels: labels of shape (batch_size, sequence_length)
    Returns:
        accuracy: average accuracy of the predictions"""

    return np.mean([pred == label for pred, label in list(zip(predictions, labels))])


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

                x = citation_normalization.normalize_citation(beam[j])
                y = citation_normalization.normalize_citation(labels[j])
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
