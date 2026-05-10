"""
Gradio helpers for the Training Results tab: load a JSON summary, render markdown,
and build matplotlib charts for dataset mix + base vs LoRA eval (focused metrics).

Charts emphasize **structured task adherence** (core schema + escalation), not clinical accuracy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Non-interactive backend for Spaces / headless servers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SUMMARY = _REPO_ROOT / "outputs/training_results_summary.json"


def summary_path() -> Path:
    raw = os.getenv("TRAINING_RESULTS_JSON", "").strip()
    return Path(raw) if raw else _DEFAULT_SUMMARY


def load_training_summary() -> dict[str, Any]:
    """Load merged training + eval summary; return empty dict if missing."""
    p = summary_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _pp_delta(base_r: float | None, lora_r: float | None) -> str:
    """Percentage-point improvement LoRA minus base."""
    if base_r is None or lora_r is None:
        return "—"
    return f"{100.0 * (float(lora_r) - float(base_r)):+.1f} pp"


def find_latest_trainer_state() -> Path | None:
    """
    Latest trainer_state.json under scale then demo adapter dirs, or TRAINER_STATE_JSON.
    Used for token / FLOPs snapshot (informative, not exact billing).
    """
    explicit = os.getenv("TRAINER_STATE_JSON", "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p.resolve()
    roots = [
        _REPO_ROOT / "outputs/mindforge-qwen-lora-scale",
        _REPO_ROOT / "outputs/mindforge-qwen-lora",
    ]
    candidates: list[Path] = []
    for r in roots:
        if r.is_dir():
            candidates.extend(r.glob("**/trainer_state.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def format_training_compute_section() -> str:
    """Markdown block from Hugging Face Trainer state (tokens logged, FLOPs estimate, loss)."""
    p = find_latest_trainer_state()
    if p is None:
        return ""

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    last_train: dict[str, Any] | None = None
    for row in reversed(data.get("log_history") or []):
        if isinstance(row, dict) and "loss" in row and "eval_loss" not in row:
            last_train = row
            break

    lines = [
        "## Training compute snapshot",
        "",
        f"- **Source:** `{p.relative_to(_REPO_ROOT)}`",
        f"- **Global step / epoch:** {data.get('global_step')} / {data.get('epoch')}",
    ]
    if last_train:
        if last_train.get("loss") is not None:
            lines.append(f"- **Last logged train loss:** {float(last_train['loss']):.4f}")
        if last_train.get("num_tokens") is not None:
            lines.append(f"- **Trainer-logged token counter (last step):** {int(last_train['num_tokens']):,}")
        if last_train.get("mean_token_accuracy") is not None:
            lines.append(
                f"- **Mean token accuracy (last log):** {float(last_train['mean_token_accuracy']):.3f}"
            )

    flos = data.get("total_flos")
    if isinstance(flos, (int, float)) and flos > 0:
        lines.append(f"- **Approx. cumulative FLOPs (Transformers estimate):** `{flos:.3e}`")

    lines += [
        "",
        "_FLOPs are library-reported estimates for relative scale, not vendor billing. "
        "For **AMD GPU utilization**, run **`rocm-smi`** or **`amd-smi monitor`** during training "
        "and save a screenshot or log for judges._",
        "",
    ]
    return "\n".join(lines)


def format_results_markdown(data: dict[str, Any]) -> str:
    """Human-readable summary for the Results tab."""
    if not data:
        p = summary_path()
        return (
            "## Training results\n\n"
            f"No summary file at `{p}`.\n\n"
            "Generate it from the repo root:\n\n"
            "```bash\npython scripts/build_training_results_summary.py\n```\n\n"
            "After you run `evaluate_base_vs_lora.py` for the scale adapter, run the "
            "builder again so charts include **Base vs +LoRA** metrics."
        )

    lines: list[str] = [
        "## What we trained for",
        data.get("training_objective", ""),
        "",
        "## Methodology",
    ]
    for bullet in data.get("methodology") or []:
        lines.append(f"- {bullet}")
    lines.append("")

    sd = data.get("scale_dataset") or {}
    lines += [
        "## 100× (scale) dataset",
        f"- **Train rows:** {sd.get('train_rows', '—')} · **Val rows:** {sd.get('validation_rows', '—')}",
        f"- **Files:** `{sd.get('train_file')}`, `{sd.get('validation_file')}`",
    ]
    note = sd.get("notes")
    if note:
        lines.append(f"- {note}")
    lines.append("")

    dd = data.get("demo_dataset") or {}
    lines += [
        "## Demo (smaller) dataset",
        f"- **Train rows:** {dd.get('train_rows', '—')} · **Val rows:** {dd.get('validation_rows', '—')}",
        "",
    ]

    on = data.get("overnight_run") or {}
    lines += [
        "## Overnight run defaults",
        f"- **Script:** `{on.get('script')}`",
        f"- **Base model:** `{on.get('default_base_model')}`",
        f"- **Adapter output:** `{on.get('default_output_dir')}`",
        f"- **Epochs / LR / max_length:** {on.get('default_epochs')} / {on.get('default_learning_rate')} / {on.get('default_max_length')}",
        "",
    ]

    lines.append(format_training_compute_section())

    ev = data.get("eval") or {}
    primary_key = ev.get("primary")
    if primary_key and ev.get(primary_key):
        block = ev[primary_key]
        m = block.get("metrics") or {}
        n_ev = m.get("n_cases")
        sc = m.get("schema_core") or {}
        es = m.get("escalation_match") or {}
        br_sc, lr_sc = sc.get("base_rate"), sc.get("lora_rate")
        br_es, lr_es = es.get("base_rate"), es.get("lora_rate")

        lines += [
            f"## Offline eval — structured adherence ({block.get('label', primary_key)})",
            "",
            "**Focus:** whether the adapter improves **care-loop JSON shape** (core fields + escalation), "
            "not clinical correctness.",
            "",
            f"- **Base model:** `{block.get('base_model')}`",
            f"- **Adapter:** `{block.get('adapter_dir')}`",
            f"- **Cases file:** `{block.get('cases_file')}` (held-out; gold labels **not** in prompt)",
            f"- **Cases run:** **{n_ev}**",
            "",
            "| Metric | Base | + LoRA | Δ (LoRA − Base) |",
            "|--------|------|--------|------------------|",
            f"| **Core schema** (relaxed contract) | {_pct(br_sc)} | {_pct(lr_sc)} | {_pp_delta(br_sc, lr_sc)} |",
            f"| **Escalation** (gold, heuristic) | {_pct(br_es)} | {_pct(lr_es)} | {_pp_delta(br_es, lr_es)} |",
            "",
        ]
    else:
        lines += [
            "## Offline eval",
            "_No eval report merged yet._ Run `evaluate_base_vs_lora.py`, then "
            "`python scripts/build_training_results_summary.py`.",
            "",
        ]

    lines.append("### Refresh")
    lines.append(
        "Use **Refresh results** after rebuilding the summary, or restart the app if you replaced the JSON on disk."
    )
    return "\n".join(lines)


def figure_dataset_risk_distribution(data: dict[str, Any]) -> plt.Figure | None:
    """Horizontal bar chart of scale train set risk_level counts."""
    sd = data.get("scale_dataset") or {}
    dist = sd.get("risk_distribution_train") or {}
    if not dist:
        return None
    order = ["LOW", "MODERATE", "HIGH", "CRISIS"]
    labels = [k for k in order if k in dist]
    if not labels:
        labels = list(dist.keys())
    values = [int(dist[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=125)
    y = np.arange(len(labels))
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(labels)))
    ax.barh(y, values, color=cmap, height=0.65)
    ax.set_yticks(y, labels, fontsize=11)
    ax.set_xlabel("Rows (scale train, cleaned)", fontsize=11)
    ax.set_title("Training data mix by risk_level", fontsize=12, fontweight="bold")
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.012, i, str(v), va="center", fontsize=10)
    ax.set_xlim(0, max(values) * 1.15)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.14)
    return fig


def figure_base_vs_lora(data: dict[str, Any]) -> plt.Figure | None:
    """
    Two-metric chart: core schema + escalation only, with value labels and Δ annotations.
    """
    ev = data.get("eval") or {}
    key = ev.get("primary")
    if not key or not ev.get(key):
        return None
    m = ev[key].get("metrics") or {}
    sc = m.get("schema_core") or {}
    es = m.get("escalation_match") or {}

    names_display = ["Core schema\n(relaxed)", "Escalation match\n(gold, heuristic)"]
    base_rates = [sc.get("base_rate"), es.get("base_rate")]
    lora_rates = [sc.get("lora_rate"), es.get("lora_rate")]
    if all(x is None for x in base_rates + lora_rates):
        return None

    def to_pct(x: Any) -> float:
        return float(x) * 100.0 if x is not None else 0.0

    b = [to_pct(x) for x in base_rates]
    l = [to_pct(x) for x in lora_rates]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=128)
    bars_b = ax.bar(x - width / 2, b, width, label="Base only", color="#64748b", edgecolor="#475569", linewidth=0.8)
    bars_l = ax.bar(x + width / 2, l, width, label="Base + MindForge LoRA", color="#8b5cf6", edgecolor="#6d28d9", linewidth=0.8)

    # Bar value labels + delta badges (percentage points).
    for i in range(2):
        ax.text(
            i - width / 2,
            b[i] + 2.5,
            f"{b[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#e2e8f0",
        )
        ax.text(
            i + width / 2,
            l[i] + 2.5,
            f"{l[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#ede9fe",
        )
        delta = l[i] - b[i]
        if abs(delta) > 0.05:
            ax.annotate(
                f"Δ {delta:+.0f} pp",
                xy=(i, max(b[i], l[i])),
                xytext=(i, min(100, max(b[i], l[i]) + 18)),
                ha="center",
                fontsize=9,
                color="#4ade80" if delta > 0 else "#f87171",
                fontweight="bold",
            )

    ax.set_ylabel("Pass rate (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_xticks(x, names_display, fontsize=10)
    ax.axhline(100, color="#334155", linewidth=0.7, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    title = ev[key].get("label", "Scale adapter")
    ax.set_title(
        f"Structured task adherence — {title}\n"
        "(higher is better · same held-out eval cases · no gold in prompt)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.18)
    return fig


def _placeholder_fig(title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=110)
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=11)
    ax.text(
        0.5,
        0.28,
        "python scripts/build_training_results_summary.py",
        ha="center",
        va="center",
        fontsize=9,
        color="#94a3b8",
    )
    ax.axis("off")
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)
    return fig


def refresh_results_panel() -> tuple[str, plt.Figure, plt.Figure]:
    """Reload JSON from disk and rebuild markdown + figures (Gradio button callback)."""
    data = load_training_summary()
    md = format_results_markdown(data)
    f_dist = figure_dataset_risk_distribution(data)
    if f_dist is None:
        f_dist = _placeholder_fig("Scale risk distribution unavailable.\nRun build_training_results_summary.py.")
    f_cmp = figure_base_vs_lora(data)
    if f_cmp is None:
        f_cmp = _placeholder_fig(
            "Base vs LoRA chart appears after you run evaluate_base_vs_lora.py\n"
            "and rebuild the summary (see Training results tab instructions)."
        )
    return md, f_dist, f_cmp
