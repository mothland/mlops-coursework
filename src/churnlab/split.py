"""Reproducible stratified train/validation/test split (60/20/20)."""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from churnlab.data import ID_COLUMN, TARGET


def split_data(
    df: pd.DataFrame, seed: int, valid_size: float = 0.2, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split once with one seed. Stratified on the raw Churn label at both stages."""
    train_df, remaining_df = train_test_split(
        df, test_size=valid_size + test_size, stratify=df[TARGET], random_state=seed
    )
    valid_df, test_df = train_test_split(
        remaining_df,
        test_size=test_size / (valid_size + test_size),
        stratify=remaining_df[TARGET],
        random_state=seed,
    )
    return train_df, valid_df, test_df


def allocation_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """customerID -> split name, sorted by ID so the table is stable across runs."""
    rows = [
        pd.DataFrame({ID_COLUMN: part[ID_COLUMN].to_numpy(), "split": name})
        for name, part in splits.items()
    ]
    return pd.concat(rows).sort_values(ID_COLUMN).reset_index(drop=True)


def allocation_hash(allocation: pd.DataFrame) -> str:
    return hashlib.sha256(allocation.to_csv(index=False).encode("utf-8")).hexdigest()


def assert_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for name, part in splits.items():
        for customer in part[ID_COLUMN]:
            if customer in seen:
                raise ValueError(f"{customer} is in both {seen[customer]} and {name}")
            seen[customer] = name
