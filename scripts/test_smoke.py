"""CPU-only smoke tests — run without a GPU (no torch/unsloth/trl import).

These guard the lab source against the most common breakages so `make test`
is a real gate, not a no-op:
- every notebook/script file exists and is valid Python (catches syntax errors)
- the TRL trainer calls use `processing_class=` (TRL >= 0.13), NOT the removed
  `tokenizer=` arg — the regression that broke NB1/NB3 on the resolved trl 0.19.x

Run:  pytest -q scripts/   (or `make test`).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "01_sft_mini", "02_preference_data", "03_dpo_train",
    "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
]


def test_notebooks_exist_and_parse():
    for nb in NOTEBOOKS:
        p = REPO / "notebooks" / f"{nb}.py"
        assert p.exists(), f"missing notebook {p}"
        ast.parse(p.read_text(encoding="utf-8"))  # SyntaxError if broken


def test_scripts_parse():
    for p in (REPO / "scripts").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8"))


def test_colab_notebooks_are_valid_json():
    for p in (REPO / "colab").glob("*.ipynb"):
        json.loads(p.read_text(encoding="utf-8"))  # ValueError if corrupt


def test_colab_notebooks_match_sources():
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_colab.py"), "--check"],
        cwd=REPO,
        check=True,
    )


def test_sft_uses_available_vietnamese_alpaca_dataset_and_columns():
    notebook = (REPO / "notebooks/01_sft_mini.py").read_text(encoding="utf-8")
    assert "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated" in notebook
    assert 'row.get("instruction_vi")' in notebook
    assert 'row.get("input_vi")' in notebook
    assert 'row.get("output_vi")' in notebook
    assert "5CD-AI/Vietnamese-alpaca-cleaned" not in notebook


def test_documented_environment_overrides_are_wired_into_sources():
    notebook_sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (REPO / "notebooks").glob("*.py")
    }
    model_consumers = {
        "01_sft_mini.py",
        "03_dpo_train.py",
        "04_compare_and_eval.py",
        "05_merge_deploy_gguf.py",
        "06_benchmark.py",
    }
    for name in model_consumers:
        assert 'os.environ.get("BASE_MODEL"' in notebook_sources[name]

    preference_source = notebook_sources["02_preference_data.py"]
    assert 'os.environ.get("PREF_SLICE"' in preference_source
    assert "DEFAULT_PREF_SLICE = 2000" in preference_source


def test_trainer_uses_processing_class_not_tokenizer():
    # TRL >= 0.13 removed the `tokenizer=` arg in favour of `processing_class=`.
    # With the requirements pin `trl>=0.12,<0.20` a fresh install resolves to
    # 0.19.x, where `DPOTrainer/SFTTrainer(tokenizer=...)` raises TypeError.
    targets = [
        "notebooks/01_sft_mini.py",
        "notebooks/03_dpo_train.py",
        "scripts/train_dpo.py",
        "colab/Lab22_DPO_T4.ipynb",
        "colab/Lab22_DPO_BigGPU.ipynb",
    ]
    offenders = [t for t in targets if "tokenizer=tokenizer" in (REPO / t).read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} still pass tokenizer=tokenizer to a TRL trainer; "
        f"use processing_class=tokenizer (tokenizer= removed in trl>=0.13)."
    )


def test_gguf_export_loads_final_dpo_adapter():
    notebook = (REPO / "notebooks/05_merge_deploy_gguf.py").read_text(encoding="utf-8")
    cli = (REPO / "scripts/merge_and_gguf.py").read_text(encoding="utf-8")
    assert "PeftModel.from_pretrained(model, str(DPO_PATH))" in notebook
    assert "PeftModel.from_pretrained(model, args.dpo_path)" in cli
    assert "PeftModel.from_pretrained(model, str(SFT_PATH))" not in notebook


def test_colab_run_order_and_no_embedded_secret():
    for path in (REPO / "colab").glob("*.ipynb"):
        text = path.read_text(encoding="utf-8")
        assert text.index("notebooks/04_compare_and_eval.py") < text.index("Bonus — β-sweep")
        assert text.index("Bonus — β-sweep") < text.index("notebooks/06_benchmark.py")
        assert text.index("notebooks/06_benchmark.py") < text.index("notebooks/05_merge_deploy_gguf.py")
        assert "userdata.get('XAH_API_KEY')" in text
        assert "Authorization: Bearer sk-" not in text
