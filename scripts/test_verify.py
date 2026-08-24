"""CPU-only regression tests for malformed submission artifacts."""
from __future__ import annotations

import json

from scripts.verify import check_dpo_metrics, check_manual_judgments


def test_check_dpo_metrics_reports_wrong_schema_without_crashing(tmp_path):
    metrics_path = tmp_path / "adapters" / "dpo" / "dpo_metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    problems = []

    assert check_dpo_metrics(tmp_path, problems) is False
    assert problems and problems[0].startswith("CORRUPT")


def test_check_dpo_metrics_rejects_non_finite_gap(tmp_path):
    metrics_path = tmp_path / "adapters" / "dpo" / "dpo_metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text('{"end_reward_gap": "NaN"}', encoding="utf-8")
    problems = []

    assert check_dpo_metrics(tmp_path, problems) is False
    assert "must be finite" in problems[0]


def test_check_manual_judgments_reports_wrong_row_schema(tmp_path):
    results_path = tmp_path / "data" / "eval" / "judge_results.json"
    results_path.parent.mkdir(parents=True)
    results_path.write_text(json.dumps(["not-an-object"] * 8), encoding="utf-8")
    problems = []

    assert check_manual_judgments(tmp_path, problems) is False
    assert problems and problems[0].startswith("CORRUPT")
