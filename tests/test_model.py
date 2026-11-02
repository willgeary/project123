import pickle

import numpy as np
import pandas as pd

from assignment.features import FEATURES
from assignment.model import predict, train


def test_artifact_is_plain_and_round_trips():
    """The saved model holds only NumPy arrays, survives pickling, and returns valid probabilities."""
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES).assign(y=rng.integers(0, 2, 200))
    artifact = pickle.loads(pickle.dumps(train(panel)))
    assert all(isinstance(artifact[k], np.ndarray) for k in ("mu", "sd"))  # no pandas objects: loads without pyarrow
    p = predict(artifact, panel)
    assert p.shape == (200,) and ((p > 0) & (p < 1)).all()
