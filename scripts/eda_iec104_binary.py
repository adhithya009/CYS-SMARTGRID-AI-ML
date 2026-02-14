import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_ROOT = r"\\?\C:\Users\mgiri\OneDrive\Desktop\Hemanth's academic Stuff\College\Amrita\Machine learnign(ML)\datasets\datasets\All the datasets"
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
TARGET_SCHEMA_COLS = 84
EXCLUDE_DIR_TOKEN = "Balanced_IEC104_Train_Test_CSV_Files"
MAX_SAMPLE_ROWS_FOR_PLOTS = 120_000
TOP_CORR_FEATURES = 25
TOP_DIST_FEATURES = 6


def list_csv_files(root: str) -> list[str]:
    files = []
    for dp, _, fn in os.walk(root):
        for name in fn:
            if name.lower().endswith(".csv"):
                files.append(os.path.join(dp, name))
    files.sort()
    return files


def save_class_balance_plot(labels_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels_df["label_name"], labels_df["count"], color=["#4E79A7", "#E15759"])
    ax.set_title("Binary Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Row Count")
    for bar, pct in zip(bars, labels_df["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution_binary.png", dpi=150)
    plt.close(fig)


def save_missingness_plot(missing_df: pd.DataFrame) -> None:
    top = missing_df.head(20).sort_values("missing_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["column"], top["missing_pct"], color="#76B7B2")
    ax.set_title("Top 20 Columns by Missing Percentage")
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("Column")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "missingness_top20.png", dpi=150)
    plt.close(fig)


def save_zero_pct_plot(numeric_df: pd.DataFrame) -> None:
    top = numeric_df.head(20).sort_values("zero_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["column"], top["zero_pct"], color="#59A14F")
    ax.set_title("Top 20 Numeric Columns by Zero Percentage")
    ax.set_xlabel("Zero Values (%)")
    ax.set_ylabel("Column")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "zero_pct_top20.png", dpi=150)
    plt.close(fig)


def save_inf_count_plot(numeric_df: pd.DataFrame) -> None:
    top = numeric_df[numeric_df["inf_count"] > 0].sort_values("inf_count", ascending=False).head(20)
    if top.empty:
        return
    top = top.sort_values("inf_count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["column"], top["inf_count"], color="#F28E2B")
    ax.set_title("Columns with Infinite Values")
    ax.set_xlabel("Infinite Value Count")
    ax.set_ylabel("Column")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "infinite_values_top20.png", dpi=150)
    plt.close(fig)


def save_file_rows_plot(file_rows: list[tuple[str, int, int]]) -> None:
    rows = [x[1] for x in file_rows if x[1] > 0]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rows, bins=30, color="#EDC948", edgecolor="black", alpha=0.85)
    ax.set_title("Distribution of Rows per Selected CSV File")
    ax.set_xlabel("Rows per File")
    ax.set_ylabel("Number of Files")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "rows_per_file_hist.png", dpi=150)
    plt.close(fig)


def save_correlation_heatmap(sample_df: pd.DataFrame) -> None:
    numeric_cols = [c for c in sample_df.columns if c != "label_bin"]
    if not numeric_cols:
        return

    variances = sample_df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
    top_cols = variances.head(TOP_CORR_FEATURES).index.tolist()
    if not top_cols:
        return

    corr = sample_df[top_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(f"Correlation Heatmap (Top {len(top_cols)} Variance Features)")
    ax.set_xticks(np.arange(len(top_cols)))
    ax.set_yticks(np.arange(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=90, fontsize=8)
    ax.set_yticklabels(top_cols, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap_top25.png", dpi=150)
    plt.close(fig)


def save_top_feature_distributions(sample_df: pd.DataFrame) -> None:
    if "label_bin" not in sample_df.columns:
        return

    numeric_cols = [c for c in sample_df.columns if c != "label_bin"]
    if not numeric_cols:
        return

    corr_to_label = sample_df[numeric_cols].corrwith(sample_df["label_bin"]).abs().sort_values(ascending=False)
    top_cols = corr_to_label.head(TOP_DIST_FEATURES).index.tolist()
    if not top_cols:
        return

    n = len(top_cols)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, max(3 * n, 8)))
    if n == 1:
        axes = [axes]

    normal = sample_df[sample_df["label_bin"] == 0]
    attack = sample_df[sample_df["label_bin"] == 1]

    for ax, col in zip(axes, top_cols):
        n_vals = pd.to_numeric(normal[col], errors="coerce").dropna().to_numpy()
        a_vals = pd.to_numeric(attack[col], errors="coerce").dropna().to_numpy()
        if n_vals.size == 0 or a_vals.size == 0:
            continue
        ax.hist(n_vals, bins=60, density=True, alpha=0.5, color="#4E79A7", label="NORMAL (0)")
        ax.hist(a_vals, bins=60, density=True, alpha=0.5, color="#E15759", label="ATTACK (1)")
        ax.set_title(f"{col} distribution by class")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Feature Value")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top_features_distribution_by_class.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_csv = list_csv_files(DATA_ROOT)
    schema_files = []
    skipped_schema = 0
    skipped_excluded = 0
    header_errors = []

    for path in all_csv:
        if EXCLUDE_DIR_TOKEN.lower() in path.lower():
            skipped_excluded += 1
            continue
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
            if len(cols) == TARGET_SCHEMA_COLS:
                schema_files.append(path)
            else:
                skipped_schema += 1
        except Exception as exc:
            header_errors.append((path, str(exc)))

    row_count = 0
    duplicate_count = 0
    label_counter = Counter()
    missing_counter = Counter()
    nonmissing_counter = Counter()
    inf_counter = Counter()
    numeric_values = {}
    file_rows = []
    file_errors = []
    numeric_cols = None
    sample_frames = []
    sampled_rows_so_far = 0
    rng = np.random.default_rng(42)

    for path in schema_files:
        file_rows_local = 0
        file_dups_local = 0
        try:
            for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
                file_rows_local += len(chunk)
                row_count += len(chunk)
                file_dups_local += int(chunk.duplicated().sum())

                if "Label" not in chunk.columns:
                    continue

                bin_label = np.where(chunk["Label"].astype("string") == "NORMAL", 0, 1)
                vals, counts = np.unique(bin_label, return_counts=True)
                for v, c in zip(vals, counts):
                    label_counter[int(v)] += int(c)

                missing = chunk.isna().sum()
                for col, miss in missing.items():
                    miss_i = int(miss)
                    missing_counter[col] += miss_i
                    nonmissing_counter[col] += int(len(chunk) - miss_i)

                if numeric_cols is None:
                    numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
                    for col in numeric_cols:
                        numeric_values[col] = []

                for col in numeric_cols:
                    s = pd.to_numeric(chunk[col], errors="coerce")
                    finite_mask = np.isfinite(s)
                    inf_counter[col] += int((~finite_mask & s.notna()).sum())
                    numeric_values[col].append(s[finite_mask].to_numpy())

                if sampled_rows_so_far < MAX_SAMPLE_ROWS_FOR_PLOTS and numeric_cols:
                    remaining = MAX_SAMPLE_ROWS_FOR_PLOTS - sampled_rows_so_far
                    sample_n = min(1500, len(chunk), remaining)
                    if sample_n > 0:
                        take_idx = rng.choice(len(chunk), size=sample_n, replace=False)
                        sample_chunk = chunk.iloc[take_idx][numeric_cols].copy()
                        sample_chunk = sample_chunk.apply(pd.to_numeric, errors="coerce")
                        sample_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                        sample_chunk.dropna(how="all", inplace=True)
                        if not sample_chunk.empty:
                            sample_chunk["label_bin"] = np.where(
                                chunk.iloc[take_idx]["Label"].astype("string") == "NORMAL", 0, 1
                            )
                            sample_frames.append(sample_chunk)
                            sampled_rows_so_far += len(sample_chunk)

            duplicate_count += file_dups_local
            file_rows.append((path, file_rows_local, file_dups_local))
        except Exception as exc:
            file_errors.append((path, str(exc)))

    missing_rows = []
    for col in sorted(set(missing_counter.keys()) | set(nonmissing_counter.keys())):
        denom = missing_counter[col] + nonmissing_counter[col]
        missing_pct = (100.0 * missing_counter[col] / denom) if denom else 0.0
        missing_rows.append({"column": col, "missing_pct": round(missing_pct, 6)})
    missing_df = pd.DataFrame(missing_rows).sort_values("missing_pct", ascending=False)
    missing_df.to_csv(OUTPUT_DIR / "missingness_84col.csv", index=False)

    numeric_rows = []
    for col, parts in numeric_values.items():
        if not parts:
            continue
        arr = np.concatenate(parts)
        if arr.size == 0:
            continue
        zeros = int((arr == 0).sum())
        numeric_rows.append(
            {
                "column": col,
                "count": int(arr.size),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "zero_pct": round(100.0 * zeros / arr.size, 6),
                "inf_count": int(inf_counter[col]),
            }
        )
    numeric_df = pd.DataFrame(numeric_rows).sort_values("zero_pct", ascending=False)
    numeric_df.to_csv(OUTPUT_DIR / "numeric_quality_84col.csv", index=False)

    labels_df = pd.DataFrame(
        [
            {"label_bin": 0, "label_name": "NORMAL", "count": int(label_counter[0])},
            {"label_bin": 1, "label_name": "ATTACK", "count": int(label_counter[1])},
        ]
    )
    labels_df["pct"] = (labels_df["count"] / labels_df["count"].sum() * 100).round(6)
    labels_df.to_csv(OUTPUT_DIR / "label_distribution_binary_84col.csv", index=False)

    file_rows_df = pd.DataFrame(file_rows, columns=["path", "rows", "dups"])
    file_rows_df.to_csv(OUTPUT_DIR / "rows_per_file_84col.csv", index=False)

    save_class_balance_plot(labels_df)
    save_missingness_plot(missing_df)
    save_zero_pct_plot(numeric_df)
    save_inf_count_plot(numeric_df)
    save_file_rows_plot(file_rows)
    if sample_frames:
        sample_df = pd.concat(sample_frames, ignore_index=True)
        sample_df.dropna(axis=1, how="all", inplace=True)
        if "label_bin" in sample_df.columns:
            sample_df["label_bin"] = pd.to_numeric(sample_df["label_bin"], errors="coerce").fillna(0).astype(int)
            save_correlation_heatmap(sample_df)
            save_top_feature_distributions(sample_df)

    largest_files = sorted(file_rows, key=lambda x: x[1], reverse=True)[:10]
    dup_files = [x for x in sorted(file_rows, key=lambda x: x[2], reverse=True)[:10] if x[2] > 0]

    summary = {
        "dataset_root": DATA_ROOT,
        "all_csv_files_found": len(all_csv),
        "selected_84col_files": len(schema_files),
        "skipped_excluded_files": skipped_excluded,
        "skipped_non_84col_files": skipped_schema,
        "header_errors": len(header_errors),
        "read_errors": len(file_errors),
        "total_rows_selected": row_count,
        "total_duplicate_rows_selected": duplicate_count,
        "binary_label_counts": {"NORMAL_0": int(label_counter[0]), "ATTACK_1": int(label_counter[1])},
        "binary_attack_rate_pct": round((100.0 * label_counter[1] / (label_counter[0] + label_counter[1])), 6),
        "top_10_largest_files": largest_files,
        "top_10_files_with_duplicates": dup_files,
    }

    with open(OUTPUT_DIR / "eda_summary_binary_84col.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "# IEC104 EDA (84-column schema, binary target)",
        "",
        f"- Total CSV files found: {summary['all_csv_files_found']}",
        f"- Selected files (84 columns, excluding `{EXCLUDE_DIR_TOKEN}`): {summary['selected_84col_files']}",
        f"- Rows in selected files: {summary['total_rows_selected']}",
        f"- Duplicate rows in selected files (within-file): {summary['total_duplicate_rows_selected']}",
        f"- NORMAL (0): {summary['binary_label_counts']['NORMAL_0']}",
        f"- ATTACK (1): {summary['binary_label_counts']['ATTACK_1']}",
        f"- Attack rate: {summary['binary_attack_rate_pct']}%",
        f"- Header errors: {summary['header_errors']}",
        f"- Read errors: {summary['read_errors']}",
        "",
        "## Output files",
        "- `outputs/eda_summary_binary_84col.json`",
        "- `outputs/label_distribution_binary_84col.csv`",
        "- `outputs/missingness_84col.csv`",
        "- `outputs/numeric_quality_84col.csv`",
        "- `outputs/rows_per_file_84col.csv`",
        "",
        "## Plots",
        "- `outputs/plots/class_distribution_binary.png`",
        "- `outputs/plots/missingness_top20.png`",
        "- `outputs/plots/zero_pct_top20.png`",
        "- `outputs/plots/infinite_values_top20.png` (if any infinite values exist)",
        "- `outputs/plots/rows_per_file_hist.png`",
        "- `outputs/plots/correlation_heatmap_top25.png`",
        "- `outputs/plots/top_features_distribution_by_class.png`",
    ]
    (OUTPUT_DIR / "EDA_REPORT_BINARY_84COL.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("EDA completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
