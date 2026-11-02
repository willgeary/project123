import numpy as np

from assignment.evaluation import flag_top, metrics


def test_top_k_metrics():
    """Top-k flagging, precision, recall and accuracy match a hand-computed example."""
    y = np.array([1, 0, 1, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.05])
    assert flag_top(p, 3).tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    m = metrics(y, p, 3)
    assert m["Precision"] == 2 / 3 and m["Recall (surgeries caught on the list)"] == 2 / 3 and m["Accuracy"] == 6 / 8
