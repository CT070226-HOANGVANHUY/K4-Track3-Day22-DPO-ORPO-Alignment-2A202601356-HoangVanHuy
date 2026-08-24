#!/usr/bin/env python3
"""Pre-submission sanity check + smoke mode.

Run from repo root: `make verify` (or `python scripts/verify.py`).
For a quick smoke run before training: `python scripts/verify.py --smoke`.

Exits 0 if every required artifact is present + REFLECTION.md edited beyond the
template. Exits non-zero with a checklist of what's missing — no files written.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    # Windows cp1252 consoles cannot encode the checkmark/info glyphs below.
    sys.stdout.reconfigure(errors="backslashreplace")

TEMPLATE_MARKERS = [
    r"<Họ Tên>",
    r"<A20-K1 / A20-K2",
    r"<YYYY-MM-DD>",
    r"_Answer here\.\s*≥",
    r"_Answer here\._?\s*$",
    r"<e\.g\., Free Colab T4 16GB",
]

ARTIFACT_ERRORS = (
    OSError,
    UnicodeError,
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
)


def check_file(path: Path, label: str, problems: list[str]) -> bool:
    if not path.exists():
        problems.append(f"MISSING  {label}: {path.relative_to(Path.cwd())}")
        return False
    if path.stat().st_size == 0:
        problems.append(f"EMPTY    {label}: {path.relative_to(Path.cwd())}")
        return False
    return True


def check_screenshots(folder: Path, min_count: int, problems: list[str]) -> int:
    if not folder.exists():
        problems.append("MISSING  submission/screenshots/ folder")
        return 0
    images = [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if len(images) < min_count:
        problems.append(
            f"TOO FEW  submission/screenshots/: have {len(images)}, need at least {min_count}. "
            f"See submission/screenshots/README.md for the list."
        )
    return len(images)


def check_reflection_edited(path: Path, problems: list[str]) -> bool:
    if not path.exists():
        problems.append("MISSING  submission/REFLECTION.md")
        return False
    text = path.read_text(encoding="utf-8")
    leftover = []
    for pattern in TEMPLATE_MARKERS:
        flags = re.MULTILINE if pattern.startswith("^") else 0
        if re.search(pattern, text, flags):
            leftover.append(pattern)
    if len(leftover) >= 3:
        problems.append(
            f"UNEDITED submission/REFLECTION.md still has {len(leftover)} template placeholders. "
            f"Fill in your own numbers and answers."
        )
        return False
    return True


def check_dpo_metrics(repo: Path, problems: list[str]) -> bool:
    metrics_path = repo / "adapters" / "dpo" / "dpo_metrics.json"
    if not metrics_path.exists():
        problems.append("MISSING  adapters/dpo/dpo_metrics.json (NB3 didn't complete)")
        return False
    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        gap = m.get("end_reward_gap")
        if gap is not None:
            gap = float(gap)
            if not math.isfinite(gap):
                raise ValueError("end_reward_gap must be finite")
    except ARTIFACT_ERRORS as exc:
        problems.append(f"CORRUPT  adapters/dpo/dpo_metrics.json — {exc}")
        return False
    if gap is None:
        problems.append("WARN     adapters/dpo/dpo_metrics.json has no end_reward_gap (TRL log columns missing?)")
        return True
    if gap <= 0:
        problems.append(
            f"WARN     end_reward_gap = {gap:+.3f} (≤ 0). DPO didn't separate chosen from rejected. "
            f"Check NB3 output. (Not a hard fail — explain in REFLECTION § 3 + § 5.)"
        )
    return True


def check_gguf(repo: Path, problems: list[str]) -> bool:
    gguf_dir = repo / "gguf"
    if not gguf_dir.exists():
        problems.append("MISSING  gguf/ directory (NB5 didn't run)")
        return False
    files = list(gguf_dir.glob("*.gguf"))
    if not files:
        problems.append("MISSING  gguf/*.gguf — NB5 quantization step didn't write a file")
        return False
    big = [p for p in files if p.stat().st_size > 5 * 1024**3]
    if big:
        problems.append(
            f"OVERSIZED  GGUF files > 5 GB: {[p.name for p in big]}. "
            f"Q4_K_M should be ≤ 5 GB even on 7B."
        )
    return True


def check_manual_judgments(repo: Path, problems: list[str]) -> bool:
    path = repo / "data" / "eval" / "judge_results.json"
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        valid = {"sft", "dpo", "tie"}
        unfinished = [
            row.get("id") for row in rows
            if row.get("winner_manual") not in valid
            or not str(row.get("manual_reason", "")).strip()
            or "TODO" in str(row.get("manual_reason", ""))
        ]
    except ARTIFACT_ERRORS as exc:
        problems.append(f"CORRUPT  data/eval/judge_results.json — {exc}")
        return False
    if len(rows) != 8 or unfinished:
        problems.append(
            f"MANUAL  NB4 needs 8 completed manual judgments; unfinished IDs: {unfinished}"
        )
        return False
    return True


def check_full_bonus(repo: Path, problems: list[str]) -> None:
    if list((repo / "gguf").glob("*.gguf")):
        check_gguf(repo, problems)
    else:
        # Code-only submissions intentionally keep the multi-GB file in Drive.
        # Require machine-readable metadata plus the smoke screenshot instead.
        meta_path = repo / "data" / "eval" / "deploy_meta.json"
        smoke_path = repo / "submission" / "screenshots" / "06-gguf-smoke.png"
        if not meta_path.exists() or not smoke_path.exists():
            problems.append(
                "MISSING  GGUF file, or code-only evidence pair "
                "data/eval/deploy_meta.json + 06-gguf-smoke.png"
            )
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                size_mb = float(meta.get("gguf_size_mb", 0))
                if not meta.get("smoke_response") or not 0 < size_mb < 5 * 1024:
                    problems.append("INVALID  code-only GGUF metadata (size/smoke response)")
            except ARTIFACT_ERRORS as exc:
                problems.append(f"CORRUPT  data/eval/deploy_meta.json — {exc}")
    check_file(
        repo / "data" / "eval" / "benchmark_results.json",
        "NB6 benchmark results",
        problems,
    )

    beta_path = repo / "data" / "eval" / "beta_sweep_results.json"
    if check_file(beta_path, "beta-sweep results", problems):
        try:
            payload = json.loads(beta_path.read_text(encoding="utf-8"))
            runs = payload.get("runs", [])
            betas = {float(row["beta"]) for row in runs}
            missing_rates = [row.get("beta") for row in runs if row.get("win_rate") is None]
            unfinished = [
                (row.get("beta"), judgment.get("id"))
                for row in runs
                for judgment in row.get("judgments", [])
                if judgment.get("manual_audit")
                and (
                    judgment.get("winner_manual") not in {"sft", "dpo", "tie"}
                    or "TODO" in str(judgment.get("manual_reason", ""))
                )
            ]
            if not {0.05, 0.1, 0.5}.issubset(betas):
                problems.append(f"BETA     expected 0.05/0.1/0.5, found {sorted(betas)}")
            if missing_rates:
                problems.append(f"BETA     missing judge win-rate for beta: {missing_rates}")
            if unfinished:
                problems.append(f"BETA     unfinished manual audit rows: {unfinished}")
        except ARTIFACT_ERRORS as exc:
            problems.append(f"CORRUPT  beta_sweep_results.json — {exc}")

    alpaca_path = repo / "data" / "eval" / "alpaca_lite_judgments.json"
    if check_file(alpaca_path, "NB6 AlpacaEval-lite judgments", problems):
        try:
            judgments = json.loads(alpaca_path.read_text(encoding="utf-8"))
            audited = [j for j in judgments if "winner_manual" in j]
            incomplete = [
                j.get("id") for j in audited
                if j.get("winner_manual") not in {"sft", "dpo", "tie"}
                or "TODO" in str(j.get("manual_reason", ""))
            ]
            if len(audited) != 10 or incomplete:
                problems.append(
                    f"AUDIT    NB6 needs 10 manual audit rows; have {len(audited)}, incomplete {incomplete}"
                )
        except ARTIFACT_ERRORS as exc:
            problems.append(f"CORRUPT  alpaca_lite_judgments.json — {exc}")

    expected_shots = {
        "01-setup-gpu.png",
        "02-sft-loss.png",
        "03-dpo-reward-curves.png",
        "04-side-by-side-table.png",
        "05-judge-output.png",
        "06-gguf-smoke.png",
        "07-benchmark-comparison.png",
        "bonus-beta-sweep.png",
    }
    shot_dir = repo / "submission" / "screenshots"
    existing = {p.name for p in shot_dir.iterdir()} if shot_dir.exists() else set()
    missing = sorted(expected_shots - existing)
    if missing:
        problems.append(f"SHOTS    missing full-submission screenshots: {missing}")


def check_secrets(repo: Path, problems: list[str]) -> None:
    """Scan tracked text files without ever echoing a matching secret value."""
    try:
        output = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=repo, capture_output=True, check=True,
        ).stdout
        tracked = [repo / p.decode("utf-8") for p in output.split(b"\0") if p]
    except (OSError, subprocess.SubprocessError, UnicodeError):
        tracked = [p for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts]

    patterns = [
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"Authorization[\"']?\s*:\s*[\"']Bearer\s+[A-Za-z0-9_-]{20,}", re.IGNORECASE),
    ]
    hits = []
    for path in tracked:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gguf", ".parquet"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if any(pattern.search(text) for pattern in patterns):
            hits.append(str(path.relative_to(repo)))
    if hits:
        problems.append(f"SECRET   possible credential in tracked files: {sorted(hits)}")


def smoke_check(repo: Path) -> int:
    """Light-weight pre-training check: imports work, GPU visible, deck files present."""
    print("==> Smoke check (imports + GPU + deck files)\n")
    problems: list[str] = []

    # Imports
    try:
        import torch  # noqa: WPS433
        print(f"  ✓ torch              {torch.__version__}")
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            print(f"  ✓ CUDA               {dev.name} ({dev.total_memory / 1e9:.1f} GB)")
        else:
            problems.append("torch.cuda.is_available() == False -- DPO needs a CUDA/ROCm GPU. Use the Colab T4 path (see HARDWARE-GUIDE.md); this local smoke gate cannot pass on CPU/Mac.")
    except ImportError as exc:
        problems.append(f"torch import failed: {exc}")

    for mod in ["unsloth", "trl", "peft", "bitsandbytes", "datasets", "matplotlib"]:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except (ImportError, NotImplementedError, RuntimeError) as exc:
            # unsloth raises NotImplementedError (not ImportError) when no GPU is present.
            problems.append(f"{mod} import failed: {exc}")

    # Deck source (sibling file)
    deck = repo.parent / "day07-dpo-orpo-alignment-tu-sft-en-preference-learning.tex"
    if deck.exists():
        print(f"  ✓ deck source        {deck.name}")
    else:
        print(f"  ⚠ deck source not found at {deck} — fine if you cloned standalone")

    # Notebook source files
    nb_dir = repo / "notebooks"
    expected_nbs = [
        "01_sft_mini.py", "02_preference_data.py", "03_dpo_train.py",
        "04_compare_and_eval.py", "05_merge_deploy_gguf.py", "06_benchmark.py",
    ]
    for nb in expected_nbs:
        if (nb_dir / nb).exists():
            print(f"  ✓ {nb}")
        else:
            problems.append(f"missing notebook {nb_dir / nb}")

    # NB6 benchmark dependency check
    try:
        import lm_eval  # noqa: F401
        print("  ✓ lm_eval (NB6 benchmark suite)")
    except ImportError:
        problems.append("lm_eval missing — pip install -r requirements.txt (NB6 will fail)")

    print()
    if problems:
        print("✗ Smoke check FAILED:\n")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("✓ Smoke check passed. You can now run `make pipeline` (or open a notebook).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run pre-training smoke check (imports + GPU) instead of submission gatekeeper",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Require NB5 + NB6 + beta-sweep, manual audits, 8 screenshots, and secret scan",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent

    if args.smoke:
        return smoke_check(repo)

    problems: list[str] = []
    print(f"==> Verifying submission readiness at {repo}\n")

    # Notebook source files
    for nb in ["01_sft_mini.py", "02_preference_data.py", "03_dpo_train.py",
               "04_compare_and_eval.py", "05_merge_deploy_gguf.py"]:
        check_file(repo / "notebooks" / nb, f"notebook {nb}", problems)

    # Adapter outputs
    check_file(repo / "adapters" / "sft-mini" / "adapter_config.json",
               "SFT-mini adapter config (NB1 output)", problems)
    check_file(repo / "adapters" / "dpo" / "adapter_config.json",
               "DPO adapter config (NB3 output)", problems)

    # DPO metrics
    check_dpo_metrics(repo, problems)

    # Preference data
    check_file(repo / "data" / "pref" / "train.parquet",
               "preference data parquet (NB2 output)", problems)

    # Eval outputs
    check_file(repo / "data" / "eval" / "side_by_side.jsonl",
               "side-by-side eval (NB4 output)", problems)
    check_file(repo / "data" / "eval" / "judge_results.json",
               "judge results (NB4 output)", problems)

    # OPTIONAL (bonus) — NB5 GGUF + NB6 benchmark: report, do NOT fail core
    optional = []
    if not list((repo / "gguf").glob("*.gguf")):
        optional.append("NB5 GGUF (gguf/*.gguf) not done")
    if not (repo / "data" / "eval" / "benchmark_results.json").exists():
        optional.append("NB6 benchmark (data/eval/benchmark_results.json) not done")

    # Submission artifacts (core)
    check_reflection_edited(repo / "submission" / "REFLECTION.md", problems)
    n_shots = check_screenshots(repo / "submission" / "screenshots", min_count=3, problems=problems)
    if n_shots:
        print(f"  ✓ submission/screenshots/ has {n_shots} image(s)")

    if args.full:
        check_manual_judgments(repo, problems)
        check_full_bonus(repo, problems)
        check_secrets(repo, problems)

    if optional and not args.full:
        print("\nⓘ Optional (bonus) not done — fine for a core pass:")
        for line in optional:
            print(f"  - {line}")

    print()
    if not problems:
        label = "Full core +20 checks" if args.full else "Core checks"
        print(f"✓ {label} passed. Push your repo (public!) and paste the URL into LMS.")
        return 0

    print("✗ Submission not ready yet:\n")
    for line in problems:
        print(f"  - {line}")
    target = "make verify-full" if args.full else "make verify"
    print(f"\nFix the items above and rerun `{target}`. See rubric.md for full grading details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
