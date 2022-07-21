import train_helpers

def test_batch_means_nested():
    inputs = [{"acc": 0.5, "loss": [0.5, 8]},
            {"acc": 0.6, "loss": [0.6, 9]},]
    res = train_helpers.mean_over_metrics_batches(inputs)
    assert res["acc"] == 0.55
    assert res["loss"] == 4.525