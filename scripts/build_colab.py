#!/usr/bin/env python3
"""Regenerate the stitched T4/BigGPU Colab notebooks from Jupytext sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPO_URL = (
    "https://github.com/CT070226-HOANGVANHUY/"
    "K4-Track3-Day22-DPO-ORPO-Alignment-2A202601356-HoangVanHuy.git"
)


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def parse_percent_notebook(path: Path) -> list[dict]:
    """Parse the small subset of Jupytext percent syntax used by this lab."""
    cells: list[dict] = []
    kind: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        if kind is None:
            lines = []
            return
        text = "".join(lines)
        if kind == "markdown":
            converted = []
            for line in text.splitlines(keepends=True):
                if line.startswith("# "):
                    converted.append(line[2:])
                elif line.startswith("#"):
                    converted.append(line[1:])
                else:
                    converted.append(line)
            cells.append(markdown_cell("".join(converted)))
        else:
            cells.append(code_cell(text))
        lines = []

    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
        elif kind is not None:
            lines.append(line)
    flush()
    return cells


def setup_cells(tier: str) -> list[dict]:
    model = "Qwen2.5-3B" if tier == "T4" else "Qwen2.5-7B"
    return [
        markdown_cell(
            f"# Lab 22 — DPO/ORPO Alignment ({tier} tier)\n\n"
            "**Track 3 · Day 22 · VinUni AICB program**\n\n"
            f"Run order: NB1 → NB2 → NB3 → NB4 → β-sweep → NB6 → NB5. "
            f"Tier `{tier}` uses {model}.\n\n"
            "> Before running, push the prepared code to GitHub, select a GPU runtime, "
            "and add a fresh `XAH_API_KEY` in Colab Secrets. Never paste it into a cell.\n"
        ),
        markdown_cell("## A. Colab setup — clone repo, install dependencies, load secret\n"),
        code_cell(
            "import os\n"
            f"os.environ['COMPUTE_TIER'] = '{tier}'\n"
            "os.environ.pop('JUDGE_PROVIDER', None)\n"
            "os.environ.pop('XAH_API_KEY', None)\n"
            "os.environ['JUDGE_BASE_URL'] = 'https://api.xah.io/v1'\n"
            "os.environ['JUDGE_MODEL'] = 'rouyea98/gpt-5.4'\n"
            "try:\n"
            "    from google.colab import userdata\n"
            "    secret = userdata.get('XAH_API_KEY')\n"
            "    if secret:\n"
            "        os.environ['JUDGE_PROVIDER'] = 'xah'\n"
            "        os.environ['XAH_API_KEY'] = secret\n"
            "        print('XAH_API_KEY loaded from Colab Secrets (value hidden).')\n"
            "except Exception as exc:\n"
            "    print(f'XAH_API_KEY not loaded: {type(exc).__name__}. Manual mode remains available.')\n"
            "print(f\"COMPUTE_TIER={os.environ['COMPUTE_TIER']}  "
            "JUDGE_PROVIDER={os.environ.get('JUDGE_PROVIDER', 'manual')}\")\n"
        ),
        code_cell(
            "# Install the tested stack. Restart the runtime only if pip explicitly requests it.\n"
            "!pip install -q \\\n"
            "  \"unsloth>=2025.10,<2026.5\" \"trl>=0.12,<0.20\" \"peft>=0.13,<1.0\" \\\n"
            "  \"bitsandbytes>=0.44,<1.0\" \"datasets>=3.1,<4.0\" \"accelerate>=1.1,<2.0\" \\\n"
            "  \"llama-cpp-python>=0.3,<1.0\" \"lm-eval[ifeval,math]>=0.4.5,<1.0\" \\\n"
            "  \"matplotlib>=3.9,<4.0\" \"pandas>=2.2,<3.0\" \"pyarrow>=17,<22\" \\\n"
            "  \"openai>=1.55,<2.0\" \"pytest>=8.3,<9.0\"\n"
        ),
        code_cell(
            "import torch\n"
            "assert torch.cuda.is_available(), 'Runtime → Change runtime type → GPU'\n"
            "gpu = torch.cuda.get_device_properties(0)\n"
            "print(f'GPU: {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)')\n"
        ),
        code_cell(
            "# Work in a real clone so scripts, Makefile, and verifier are available.\n"
            "import subprocess\n"
            "from pathlib import Path\n"
            f"REPO_URL = {REPO_URL!r}\n"
            "WORK = Path('/content/lab22')\n"
            "if not (WORK / '.git').exists():\n"
            "    if WORK.exists():\n"
            "        import shutil\n"
            "        shutil.rmtree(WORK)\n"
            "    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(WORK)], check=True)\n"
            "else:\n"
            "    subprocess.run(['git', '-C', str(WORK), 'pull', '--ff-only'], check=True)\n"
            "os.chdir(WORK / 'notebooks')\n"
            "print(f'Working dir: {Path.cwd()}')\n"
        ),
        code_cell(
            "# Persist regenerable artifacts in Drive after every stage.\n"
            "import json as _json\n"
            "import shutil\n"
            "from datetime import datetime, timezone\n"
            "try:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "    BACKUP_ROOT = Path('/content/drive/MyDrive/lab22-artifacts')\n"
            "except Exception:\n"
            "    BACKUP_ROOT = Path('/content/lab22-artifacts-local')\n"
            "BACKUP_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "def backup_stage(stage, include_large=False):\n"
            "    paths = ['adapters/sft-mini', 'adapters/dpo', 'adapters/dpo-b0.05', "
            "'adapters/dpo-b0.50', 'data/pref', 'data/eval', 'submission/screenshots', "
            "'submission/REFLECTION.md']\n"
            "    paths += ['gguf'] if include_large else []\n"
            "    for rel in paths:\n"
            "        src, dst = WORK / rel, BACKUP_ROOT / rel\n"
            "        if src.is_dir():\n"
            "            shutil.copytree(src, dst, dirs_exist_ok=True)\n"
            "        elif src.is_file():\n"
            "            dst.parent.mkdir(parents=True, exist_ok=True)\n"
            "            shutil.copy2(src, dst)\n"
            "    log = {'last_stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat()}\n"
            "    (BACKUP_ROOT / 'backup-status.json').write_text(_json.dumps(log, indent=2))\n"
            "    print(f'Backed up {stage} → {BACKUP_ROOT}')\n"
            "\n"
            "def restore_artifacts():\n"
            "    for rel in ['adapters', 'data', 'submission/screenshots', 'gguf']:\n"
            "        src, dst = BACKUP_ROOT / rel, WORK / rel\n"
            "        if src.exists():\n"
            "            shutil.copytree(src, dst, dirs_exist_ok=True)\n"
            "    print('Artifacts restored from Drive.')\n"
        ),
        code_cell(
            "# Verify both a benign-shaped and safety-shaped judge request before long runs.\n"
            "import sys\n"
            "sys.path.insert(0, str(WORK))\n"
            "from scripts.judge_client import build_client, config_from_env, run_preflight\n"
            "judge_config = config_from_env()\n"
            "if judge_config:\n"
            "    for result in run_preflight(judge_config, client=build_client(judge_config)):\n"
            "        print(f\"{result['case']} preflight: OK\")\n"
            "else:\n"
            "    print('No API judge configured; NB4 can still use the manual rubric.')\n"
        ),
    ]


def build(tier: str, destination: Path, *, check: bool = False) -> bool:
    cells = setup_cells(tier)
    stage_order = [
        "01_sft_mini.py",
        "02_preference_data.py",
        "03_dpo_train.py",
        "04_compare_and_eval.py",
    ]
    for source in stage_order:
        cells.append(markdown_cell(f"---\n# Stage from `notebooks/{source}`\n---\n"))
        cells.extend(parse_percent_notebook(REPO / "notebooks" / source))
        cells.append(code_cell(f"backup_stage('{source[:3]}')\n"))

    cells.extend([
        markdown_cell(
            "---\n# Bonus — β-sweep\n\n"
            "Reuses the core β=0.1 adapter and trains only β=0.05 and β=0.5. "
            "Afterward, fill the three manual-audit rows per β in the result JSON.\n---\n"
        ),
        code_cell(
            "_previous_cwd = Path.cwd()\n"
            "os.chdir(WORK)\n"
            "try:\n"
            "    subprocess.run(['make', 'beta-sweep'], check=True)\n"
            "finally:\n"
            "    os.chdir(_previous_cwd)\n"
            "backup_stage('beta-sweep')\n"
        ),
    ])

    for source in ["06_benchmark.py", "05_merge_deploy_gguf.py"]:
        cells.append(markdown_cell(f"---\n# Stage from `notebooks/{source}`\n---\n"))
        cells.extend(parse_percent_notebook(REPO / "notebooks" / source))
        include_large = source.startswith("05")
        cells.append(code_cell(f"backup_stage('{source[:3]}', include_large={include_large})\n"))

    cells.extend([
        markdown_cell("---\n# Final submission checks\n---\n"),
        code_cell(
            "_previous_cwd = Path.cwd()\n"
            "os.chdir(WORK)\n"
            "try:\n"
            "    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'scripts'], check=True)\n"
            "    subprocess.run(['make', 'verify-full'], check=True)\n"
            "finally:\n"
            "    os.chdir(_previous_cwd)\n"
        ),
    ])
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4" if tier == "T4" else "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    rendered = json.dumps(notebook, ensure_ascii=False, indent=1)
    relative = destination.relative_to(REPO)
    if check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != rendered:
            print(f"STALE {relative}; run `python scripts/build_colab.py`")
            return False
        print(f"OK {relative} ({len(cells)} cells)")
        return True

    destination.write_text(rendered, encoding="utf-8")
    print(f"Wrote {relative} ({len(cells)} cells)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if a generated Colab notebook is stale; do not write files",
    )
    args = parser.parse_args()
    results = [
        build("T4", REPO / "colab" / "Lab22_DPO_T4.ipynb", check=args.check),
        build("BIGGPU", REPO / "colab" / "Lab22_DPO_BigGPU.ipynb", check=args.check),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
