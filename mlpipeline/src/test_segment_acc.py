import json
import pandas as pd
import numpy as np

from train_helpers import citation_segment_acc, split_citation_segments


def test_segmentize_statute():
    """
    38 U.S.C.A. §§ 5100A, 5102(a)(1) becomes
    [38 U.S.C.A. § 5100a, 38 U.S.C.A. § 5102a]
    """
    x = "38 U.S.C.A. §§ 5100A, 5102(a)(1)"
    y = ["38u.s.c.a.§5100a", "38u.s.c.a.§5102a"]
    ynew = split_citation_segments(x)
    assert ynew == y, ynew


def test_segemntize_case():
    """
    See Moreau v. Brown, 9 Vet. App. 389, 396 (1996)
    to
    [moreau v. brown, 9 vet.app. 389, 396]
    """
    x = "See Moreau v. Brown, 9 Vet. App. 389, 396 (1996)"
    y = ["moreauv.brown,9 vet.app. 389"]
    ynew = split_citation_segments(x)
    assert ynew == y, ynew


def get_examples():
    examples = []
    examples.append(
        """
        See Ennis v. Brown, 4 Vet. App. 523, 527
        Ennis v. Brown, 4 Vet. App, 523, 527 (1993)
        [ennisvbrown4vetapp523527]
        """
    )

    # both fed.cir. and f.3d found, but since fed.cir. is detected first it behaves as if its the only one.
    examples.append(
        """
        731 F.3d 1303, 1315 (Fed. Cir. 2013
        731 F.3d 1303, 1311 (Fed. Cir. 2013)
        [731 f3 1303] 	[731f313031315 fedcir 2013]
        """
    )
    examples.append("See, e.g., O'Hare v. Derwinski, 1 Vet. App. 365 (1991)")

# test_segemntize_case()
# test_segmentize_statute()

def reevaluate_table():
    preds_table_file = "../../data/test_demo_3_8a66f2801063e5ea77bc.table.json"

    # read file to dict
    with open(preds_table_file, "r") as f:
        preds_table = json.load(f)

    df = pd.DataFrame(preds_table["data"], columns=preds_table["columns"])
    print(preds_table["columns"])
    # apply new segment acc
    # because citation_segment_acc assumes batches we add [] around each row
    df["new_segment_acc"] = df.apply(
        lambda row: citation_segment_acc([row["top1 prediction"]], [row["label"]]),
        axis=1,
    )
    df["segment_label"] = df.apply(
        lambda row: split_citation_segments(row["label"]), axis=1
    )

    df["pred_label"] = df.apply(
        lambda row: split_citation_segments(row["top1 prediction"]), axis=1
    )
    df["avg_newsegacc"] = df.apply(
        lambda row: np.mean(row["new_segment_acc"]), axis=1
    )

    # average of new_segment_acc
    all_new_segment_accs = []
    series = df["new_segment_acc"]
    for row in series:
        all_new_segment_accs.extend(row)

    new_segment_acc_mean = np.array(all_new_segment_accs).mean()
    print(f"new_segment_acc_mean: {new_segment_acc_mean}")
    # average of old_segment_acc
    print(df["segment_acc"].mean())

    # show 10 examples for certain columns
    # drop columns
    df = df.drop(columns=["all_topk_predictions", "scores", "inputs"])

    # drop rows if 
    df = df[df["avg_newsegacc"] == 0]
    print(df.head())

    # open table in browser
    df.to_html("../../data/test_demo_3_8a66f2801063e5ea77bc.table.html")
    # launch file in browser from terminal
    # open -a /Applications/Google\ Chrome.app ../../data/test_demo_3_8a66f2801063e5ea77bc.table.html

reevaluate_table()

# from train_helpers import book_from_statute
# print(book_from_statute("38 C.F.R. 3.321(b)(1)"))
# print(split_citation_segments("38 C.F.R. 3.321(b)(1)"))