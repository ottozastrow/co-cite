import json

import numpy as np
import pandas as pd

from citation_normalization import normalize_citation, normalize_statute, sections_from_statute, segmentize_citation
from metrics import citation_segment_acc


def test_segmentize_statute():
    """
    38 U.S.C.A. §§ 5100A, 5102(a)(1) becomes
    [38 U.S.C.A. § 5100a, 38 U.S.C.A. § 5102a]
    """
    x = "38 U.S.C.A. §§ 5100A, 5102(a)(1)"
    y = ["38u.s.c.a.§5100a", "38u.s.c.a.§5102a"]
    ynew = normalize_citation(x, remove_subsections=False, remove_subsubsections=True)
    ynew = segmentize_citation(ynew)
    assert ynew == y, ynew


def test_segmentize_case():
    """
    See Moreau v. Brown, 9 Vet. App. 389, 396 (1996)
    to
    [moreau v. brown, 9 vet.app. 389, 396]
    """
    x = "See Moreau v. Brown, 9 Vet. App. 389, 396 (1996)"
    y = ["moreauv.brown,9 vet.app. 389"]
    ynew = normalize_citation(x, remove_subsections=False, remove_subsubsections=True)
    ynew = segmentize_citation(ynew)
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

    # both fed.cir. and f.3d found, but since fed.cir.
    # is detected first it behaves as if its the only one.
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


def get_table():
    preds_table_file = "../../data/test_demo_3_8a66f2801063e5ea77bc.table.json"
    preds_table_file = "../../data/test_demo_1_1bbfb8dd530d4b94b822.table.json"
    preds_table_file = "../../data/test_demo_1_1bbfb8dd530d4b94b822.table.json"
    preds_table_file = "../../data/test_demo_0_b0bdb1918bbd1d9003b8.table.json"
    preds_table_file = "../../data/test_demo_0_8c91d7eb9cc028d5b70f.table.json"

    # read file to dict
    with open(preds_table_file, "r") as f:
        preds_table = json.load(f)

    df = pd.DataFrame(preds_table["data"], columns=preds_table["columns"])
    # apply new segment acc
    # because citation_segment_acc assumes batches we add [] around each row
    df["new_segment_acc"] = df.apply(
        lambda row: citation_segment_acc(row["top1 prediction"], row["label"], False, True),
        axis=1,
    )

    # average of new_segment_acc
    all_new_segment_accs = []
    series = df["new_segment_acc"]
    for row in series:
        all_new_segment_accs.extend(row)

    new_segment_acc_mean = np.array(all_new_segment_accs).mean()
    print(f"new_segment_acc_mean: {new_segment_acc_mean}")
    print(df["segment_acc"].mean())

    df = df.drop(columns=["all_topk_predictions", "scores", "inputs"])
    
    return df
    # keep only rows with bad segment acc
    # df = df[df["avg_newsegacc"] == 0]

    # open table in browser
    df.to_html("../../data/test_demo_3_8a66f2801063e5ea77bc.table.html")
    # launch file in browser from terminal
    # open -a /Applications/Google\ Chrome.app ../../data/test_demo_3_8a66f2801063e5ea77bc.table.html

# reevaluate_table()

# from train_helpers import book_from_statute
# print(book_from_statute("38 C.F.R. 3.321(b)(1)"))
# print(split_citation_segments("38 C.F.R. 3.321(b)(1)"))
inputs = "38 CFR 3.156a"
# print(normalize_citations(inputs, remove_subsections=False, remove_subsubsections=True))
# print(citation_segment_acc("38 C.F.R. § 3.57","38 C.F.R. §§ 3.57, 3.315, 3.356", False, True))

def test_plot():
    df = get_table()
    import wandb
    # wandb.init(project="co-cite", tags="debug")
    from train_helpers import plot_accs_per_occurrence
    print(df.keys())
    plot_accs_per_occurrence(df, columns=["segment_acc", "segment_acc"])

test_plot()