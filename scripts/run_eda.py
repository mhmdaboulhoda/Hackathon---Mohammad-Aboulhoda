#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class DatasetSummary:
    fieldnames: List[str]
    features: List[str]
    rows: List[Dict[str, float | int | str | None]]
    missing_counts: Dict[str, int]
    duplicate_ids: List[int]
    duplicate_records: int
    label_counts: Counter


def load_dataset(path: Path) -> DatasetSummary:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row")

        raw_fieldnames = [name.strip() for name in reader.fieldnames]
        drop_fields = {name for name in raw_fieldnames if not name}
        fieldnames = [name for name in raw_fieldnames if name and name not in drop_fields]

        missing_counts = {name: 0 for name in fieldnames}
        rows: List[Dict[str, float | int | str | None]] = []
        duplicate_ids: List[int] = []
        seen_ids: set[int] = set()
        record_signatures: set[tuple] = set()
        duplicate_records = 0
        label_counts: Counter = Counter()

        for raw_row in reader:
            parsed_row: Dict[str, float | int | str | None] = {}
            for name in fieldnames:
                val = raw_row.get(name, "")
                if val is None or not str(val).strip():
                    missing_counts[name] += 1
                    parsed_row[name] = None
                    continue

                if name.lower() == "id":
                    parsed_row[name] = int(float(val))
                elif name.lower() == "diagnosis":
                    label = val.strip().upper()
                    parsed_row[name] = label
                    label_counts[label] += 1
                else:
                    try:
                        parsed_row[name] = float(val)
                    except ValueError:
                        missing_counts[name] += 1
                        parsed_row[name] = None

            rows.append(parsed_row)

            rec_id = parsed_row.get("id")
            if isinstance(rec_id, int):
                if rec_id in seen_ids:
                    duplicate_ids.append(rec_id)
                else:
                    seen_ids.add(rec_id)

            signature = tuple(parsed_row.get(name) for name in fieldnames)
            if signature in record_signatures:
                duplicate_records += 1
            else:
                record_signatures.add(signature)

    features = [name for name in fieldnames if name.lower() not in {"id", "diagnosis"}]
    return DatasetSummary(
        fieldnames=fieldnames,
        features=features,
        rows=rows,
        missing_counts=missing_counts,
        duplicate_ids=duplicate_ids,
        duplicate_records=duplicate_records,
        label_counts=label_counts,
    )


@dataclass(frozen=True)
class FeatureStats:
    count: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float


def _collect_feature_values(
    rows: Iterable[Dict[str, float | int | str | None]], feature: str
) -> List[float]:
    vals = [row[feature] for row in rows if isinstance(row.get(feature), (int, float))]
    return [float(v) for v in vals]


def compute_feature_stats(summary: DatasetSummary) -> Dict[str, FeatureStats]:
    stats: Dict[str, FeatureStats] = {}
    for feature in summary.features:
        vals = _collect_feature_values(summary.rows, feature)
        if not vals:
            continue
        stats[feature] = FeatureStats(
            count=len(vals),
            mean=statistics.fmean(vals),
            median=statistics.median(vals),
            std=statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            min_val=min(vals),
            max_val=max(vals),
        )
    return stats


def compute_feature_stats_by_label(summary: DatasetSummary) -> Dict[str, Dict[str, FeatureStats]]:
    grouped_stats: Dict[str, Dict[str, FeatureStats]] = {f: {} for f in summary.features}
    for label in summary.label_counts:
        label_rows = [row for row in summary.rows if row.get("diagnosis") == label]
        for feature in summary.features:
            vals = _collect_feature_values(label_rows, feature)
            if not vals:
                continue
            grouped_stats[feature][label] = FeatureStats(
                count=len(vals),
                mean=statistics.fmean(vals),
                median=statistics.median(vals),
                std=statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                min_val=min(vals),
                max_val=max(vals),
            )
    return grouped_stats


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    den = den_x * den_y
    if den == 0:
        return 0.0
    return num / den


def compute_feature_correlations(summary: DatasetSummary) -> Dict[str, float]:
    label_map = {"M": 1.0, "B": 0.0}
    correlations: Dict[str, float] = {}
    for feature in summary.features:
        x = []
        y = []
        for row in summary.rows:
            feat_val = row.get(feature)
            diag = row.get("diagnosis")
            if isinstance(feat_val, (int, float)) and diag in label_map:
                x.append(float(feat_val))
                y.append(label_map[diag])
        corr = pearson_corr(x, y)
        if not math.isnan(corr):
            correlations[feature] = corr
    return correlations


def compute_feature_correlation_matrix(summary: DatasetSummary) -> List[List[float]]:
    matrix: List[List[float]] = []
    for feature_i in summary.features:
        row_vals: List[float] = []
        for feature_j in summary.features:
            if feature_i == feature_j:
                row_vals.append(1.0)
                continue
            paired_x: List[float] = []
            paired_y: List[float] = []
            for record in summary.rows:
                val_i = record.get(feature_i)
                val_j = record.get(feature_j)
                if isinstance(val_i, (int, float)) and isinstance(val_j, (int, float)):
                    paired_x.append(float(val_i))
                    paired_y.append(float(val_j))
            corr = pearson_corr(paired_x, paired_y)
            if math.isnan(corr):
                corr = 0.0
            row_vals.append(corr)
        matrix.append(row_vals)
    return matrix


def _corr_to_hex(value: float) -> str:
    value = max(-1.0, min(1.0, value))
    intensity = int(abs(value) * 255)
    fade = 255 - intensity
    if value >= 0:
        return f"#ff{fade:02x}{fade:02x}"
    return f"#{fade:02x}{fade:02x}ff"


def build_heatmap_html(features: List[str], matrix: List[List[float]]) -> str:
    lines: List[str] = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html lang=\"en\">")
    lines.append("<head>")
    lines.append("  <meta charset=\"utf-8\">")
    lines.append("  <title>Feature Correlation Heatmap</title>")
    lines.append("  <style>")
    lines.append("    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; }")
    lines.append("    table { border-collapse: collapse; }")
    lines.append("    th, td { border: 1px solid #ddd; padding: 6px; text-align: center; font-size: 12px; }")
    lines.append("    th { position: sticky; top: 0; background: #fafafa; }")
    lines.append("  </style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append("  <h1>Feature Correlation Heatmap</h1>")
    lines.append(
        "  <p>Correlation coefficients (Pearson r) between numeric features. Reds indicate positive correlation, blues indicate negative.</p>"
    )
    lines.append("  <div style=\"overflow:auto; max-width: 100%;\">")
    lines.append("    <table>")
    header_row = "      <tr><th>Feature</th>" + "".join(f"<th>{name}</th>" for name in features) + "</tr>"
    lines.append(header_row)
    for feature, row_vals in zip(features, matrix):
        row_cells = []
        for value in row_vals:
            colour = _corr_to_hex(value)
            row_cells.append(f"<td style=\"background:{colour};\">{value:.2f}</td>")
        lines.append(f"      <tr><th>{feature}</th>{''.join(row_cells)}</tr>")
    lines.append("    </table>")
    lines.append("  </div>")
    lines.append("</body>")
    lines.append("</html>")
    return "\n".join(lines)


def build_markdown_report(
    summary: DatasetSummary,
    feature_stats: Dict[str, FeatureStats],
    grouped_stats: Dict[str, Dict[str, FeatureStats]],
    correlations: Dict[str, float],
) -> str:
    lines: List[str] = []
    lines.append("# Breast Cancer Wisconsin Dataset — Exploratory Data Analysis")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("")
    lines.append(f"*Generated on:* {ts}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Rows: **{len(summary.rows)}**")
    lines.append(f"- Columns (including ID & target): **{len(summary.fieldnames)}**")
    lines.append(f"- Feature columns analysed: **{len(summary.features)}**")
    lines.append(f"- Duplicate IDs detected: **{len(set(summary.duplicate_ids))}**")
    lines.append(f"- Duplicate full records: **{summary.duplicate_records}**")
    cols_with_missing = [c for c, count in summary.missing_counts.items() if count > 0]
    lines.append(f"- Columns with missing values: **{len(cols_with_missing)}**")
    lines.append("")

    lines.append("### Diagnosis distribution")
    lines.append("| Diagnosis | Count | Share |")
    lines.append("|-----------|-------|-------|")
    total = len(summary.rows)
    for label, count in sorted(summary.label_counts.items()):
        share = (count / total * 100) if total else 0.0
        label_name = {"M": "Malignant", "B": "Benign"}.get(label, label)
        lines.append(f"| {label_name} ({label}) | {count} | {share:.2f}% |")
    if cols_with_missing:
        lines.append("")
        lines.append("### Missing values by column")
        lines.append("| Column | Missing count |")
        lines.append("|--------|----------------|")
        for col in cols_with_missing:
            lines.append(f"| {col} | {summary.missing_counts[col]} |")
    else:
        lines.append("")
        lines.append("_No missing values detected across columns._")

    lines.append("")
    lines.append("## Feature summary statistics (overall)")
    lines.append("| Feature | Mean | Median | Std Dev | Min | Max |")
    lines.append("|---------|------|--------|---------|-----|-----|")
    for feature in summary.features:
        stats = feature_stats.get(feature)
        if not stats:
            continue
        lines.append(
            f"| {feature} | {stats.mean:.3f} | {stats.median:.3f} | {stats.std:.3f} | {stats.min_val:.3f} | {stats.max_val:.3f} |"
        )

    lines.append("")
    lines.append("## Feature means by diagnosis")
    lines.append("| Feature | Mean (Benign) | Mean (Malignant) | Difference (M - B) |")
    lines.append("|---------|----------------|------------------|---------------------|")
    for feature in summary.features:
        benign_stats = grouped_stats.get(feature, {}).get("B")
        malignant_stats = grouped_stats.get(feature, {}).get("M")
        if not benign_stats or not malignant_stats:
            continue
        diff = malignant_stats.mean - benign_stats.mean
        lines.append(
            f"| {feature} | {benign_stats.mean:.3f} | {malignant_stats.mean:.3f} | {diff:.3f} |"
        )

    lines.append("")
    lines.append("## Features most correlated with malignancy")
    lines.append("| Rank | Feature | Pearson r |")
    lines.append("|------|---------|-----------|")
    for idx, (feature, corr) in enumerate(
        sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10], start=1
    ):
        lines.append(f"| {idx} | {feature} | {corr:.3f} |")

    lines.append("")
    lines.append("## Key observations")
    insights: List[str] = []
    malignant_share = summary.label_counts.get("M", 0) / total * 100 if total else 0
    insights.append(
        f"Dataset is imbalanced with malignant cases representing about {malignant_share:.1f}% of samples."
    )

    if correlations:
        top_feature, top_corr = max(correlations.items(), key=lambda kv: abs(kv[1]))
        insights.append(
            f"`{top_feature}` shows the strongest linear relationship with malignancy (r ≈ {top_corr:.2f})."
        )

    diffs = []
    for feature in summary.features:
        b_stats = grouped_stats.get(feature, {}).get("B")
        m_stats = grouped_stats.get(feature, {}).get("M")
        if b_stats and m_stats:
            diffs.append((abs(m_stats.mean - b_stats.mean), feature, b_stats, m_stats))
    if diffs:
        diffs.sort(reverse=True)
        delta, feature, b_stats, m_stats = diffs[0]
        insights.append(
            f"Mean `{feature}` differs sharply between classes (benign {b_stats.mean:.2f} vs malignant {m_stats.mean:.2f})."
        )

    if not cols_with_missing:
        insights.append("No missing values detected, so minimal cleaning is required before modeling.")

    for point in insights:
        lines.append(f"- {point}")

    return "\n".join(lines) + "\n"


def run_eda(csv_path: Path, output_path: Path, heatmap_path: Path | None) -> None:
    summary = load_dataset(csv_path)
    feature_stats = compute_feature_stats(summary)
    grouped_stats = compute_feature_stats_by_label(summary)
    correlations = compute_feature_correlations(summary)
    report = build_markdown_report(summary, feature_stats, grouped_stats, correlations)
    output_path.write_text(report, encoding="utf-8")
    print(f"EDA report written to {output_path}")

    if heatmap_path is not None:
        matrix = compute_feature_correlation_matrix(summary)
        html = build_heatmap_html(summary.features, matrix)
        heatmap_path.write_text(html, encoding="utf-8")
        print(f"Correlation heatmap written to {heatmap_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA report for the breast cancer dataset")
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the Breast Cancer Wisconsin dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/eda_report.md"),
        help="Where to write the Markdown report (default: reports/eda_report.md)",
    )
    parser.add_argument(
        "--heatmap-html",
        type=Path,
        default=Path("reports/correlation_heatmap.html"),
        help="Where to write the HTML correlation heatmap (default: reports/correlation_heatmap.html)",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip generating the correlation heatmap",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path: Path | None
    if args.no_heatmap:
        heatmap_path = None
    else:
        args.heatmap_html.parent.mkdir(parents=True, exist_ok=True)
        heatmap_path = args.heatmap_html
    run_eda(args.csv_path, args.output, heatmap_path)


if __name__ == "__main__":
    main()
