"""Four-bar surprisal figures for the two presupposition tests.

Each bar is the mean critical-word surprisal across 24 habits.
Error bars are 95% confidence intervals.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SCORES = ROOT / "scores.csv"
FIGURE = ROOT / "surprisal_figure.png"


def mean_ci(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half = 1.96 * math.sqrt(variance / n)
    return mean, half


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[keys(row)].append(float(row["surprisal_bits"]))
    return groups


def draw(ax, labels, means, errors, title, ylabel):
    positions = range(len(labels))
    colors = ["#4C78A8", "#F58518", "#4C78A8", "#F58518"]
    ax.bar(positions, means, yerr=errors, color=colors, capsize=4, width=0.72)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    with SCORES.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    context = [row for row in rows if row["set"] == "context"]
    negation = [row for row in rows if row["set"] == "negation"]
    started = [row for row in rows if row["set"] == "started"]
    context_order = [
        ("stopped", "support", "stopped\nafter used to"),
        ("stopped", "clash", "stopped\nafter has never"),
        ("mentioned", "support", "mentioned\nafter used to"),
        ("mentioned", "clash", "mentioned\nafter has never"),
    ]
    negation_order = [
        ("stopped", "affirmative", "never after\nstopped"),
        ("stopped", "negated", "never after\ndid not stop"),
        ("mentioned", "affirmative", "never after\nmentioned"),
        ("mentioned", "negated", "never after\ndid not mention"),
    ]
    started_order = [
        ("started", "affirmative", "already after\nstarted"),
        ("started", "negated", "already after\ndid not start"),
        ("mentioned", "affirmative", "already after\nmentioned"),
        ("mentioned", "negated", "already after\ndid not mention"),
    ]

    context_groups = grouped(context, lambda row: (row["verb"], row["context"]))
    negation_groups = grouped(negation, lambda row: (row["verb"], row["polarity"]))
    started_groups = grouped(started, lambda row: (row["verb"], row["polarity"]))

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    context_stats = [mean_ci(context_groups[(verb, context_name)]) for verb, context_name, _ in context_order]
    draw(
        axes[0],
        [label for _, _, label in context_order],
        [stat[0] for stat in context_stats],
        [stat[1] for stat in context_stats],
        "Context test",
        "Surprisal at the verb (bits)",
    )
    negation_stats = [mean_ci(negation_groups[(verb, polarity)]) for verb, polarity, _ in negation_order]
    draw(
        axes[1],
        [label for _, _, label in negation_order],
        [stat[0] for stat in negation_stats],
        [stat[1] for stat in negation_stats],
        "Negation test: stopped",
        "Surprisal at never (bits)",
    )
    started_stats = [mean_ci(started_groups[(verb, polarity)]) for verb, polarity, _ in started_order]
    draw(
        axes[2],
        [label for _, _, label in started_order],
        [stat[0] for stat in started_stats],
        [stat[1] for stat in started_stats],
        "Negation test: started",
        "Surprisal at already (bits)",
    )
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=200)
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    main()
