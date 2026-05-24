from pathlib import Path

import numpy as np
import pytest

from mlclab.recovery import ClassBiasRepair, tensor_stats, top_k_with_scores


def test_class_bias_repair_fits_mean_logit_delta(tmp_path: Path) -> None:
    repair = ClassBiasRepair.fit(
        [
            ([2.0, 1.0, 0.0], [1.5, 1.0, 0.5]),
            ([1.0, 4.0, 3.0], [0.0, 2.5, 4.0]),
        ]
    )

    assert repair.sample_count == 2
    assert repair.bias.tolist() == pytest.approx([0.75, 0.75, -0.75])
    assert repair.apply([0.0, 0.0, 0.0]).tolist() == pytest.approx([0.75, 0.75, -0.75])

    path = tmp_path / "repair.json"
    repair.write_json(path)
    loaded = ClassBiasRepair.read_json(path)

    assert loaded.sample_count == repair.sample_count
    assert loaded.bias.tolist() == pytest.approx(repair.bias.tolist())


def test_class_bias_repair_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ClassBiasRepair.fit([([1.0, 2.0], [1.0])])


def test_top_k_with_scores_and_tensor_stats() -> None:
    logits = np.asarray([[0.1, 0.4, -0.2]])
    stats = tensor_stats(np.asarray([[[1, 2], [3, 4]]], dtype=np.float32))

    assert top_k_with_scores(logits, 2) == [
        {"rank": 1, "class_id": 1, "logit": 0.4},
        {"rank": 2, "class_id": 0, "logit": 0.1},
    ]
    assert stats == {
        "shape": [1, 2, 2],
        "dtype": "float32",
        "min": 1.0,
        "max": 4.0,
        "mean": 2.5,
        "std": pytest.approx(1.11803398875),
    }
