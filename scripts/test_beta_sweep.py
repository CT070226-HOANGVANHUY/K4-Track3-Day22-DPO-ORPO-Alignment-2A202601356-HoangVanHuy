"""CPU-only tests for beta-sweep artifact collection."""
from __future__ import annotations

import json

from scripts.eval_judge import collect_sweep_rows


def write_metrics(path, beta, gap):
    path.mkdir(parents=True)
    (path / "dpo_metrics.json").write_text(json.dumps({
        "beta": beta,
        "final_train_loss": 0.5,
        "end_reward_gap": gap,
        "end_chosen_reward": gap / 2,
        "end_rejected_reward": -gap / 2,
    }))


def test_collect_sweep_reuses_core_beta_point_one(tmp_path):
    write_metrics(tmp_path / "dpo", 0.1, 1.0)
    write_metrics(tmp_path / "dpo-b0.05", 0.05, 1.2)
    write_metrics(tmp_path / "dpo-b0.10", 0.1, 999.0)
    write_metrics(tmp_path / "dpo-b0.50", 0.5, 0.7)

    rows = collect_sweep_rows(tmp_path)

    assert [row["beta"] for row in rows] == [0.05, 0.1, 0.5]
    core = next(row for row in rows if row["beta"] == 0.1)
    assert core["reward_gap"] == 1.0
    assert core["adapter_dir"].endswith("dpo")
