import pandas as pd

from bayan.evaluation.bootstrap import bootstrap_ci


def add_length_bucket(df, text_col, bucket_col="length_bucket"):
    lengths = df[text_col].str.split().str.len()
    df = df.copy()
    df[bucket_col] = pd.cut(
        lengths, bins=[0, 5, 10, 20, float("inf")],
        labels=["short(<=5)", "medium(6-10)", "long(11-20)", "very_long(20+)"],
    )
    return df


def sliced_report(df, correct_col, slice_cols, min_slice_size=30, n_boot=2000, seed=42, alpha=0.05):
    rows = []
    for slice_col in slice_cols:
        for value, group in df.groupby(slice_col, observed=True):
            values = group[correct_col].to_numpy()
            n = len(values)
            point, lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed, alpha=alpha)
            rows.append({
                "slice": slice_col,
                "value": value,
                "n": n,
                "metric":
