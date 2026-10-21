"""Loading, validation and stateless parsing of the Telco CSV.

Everything here is a boundary responsibility: it runs the same way at training and at
inference and learns nothing from the data. Learned preprocessing lives in the Pipeline.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/telco.csv")
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

ID_COLUMN = "customerID"
TARGET = "Churn"
POSITIVE_LABEL = "Yes"  # Churn = Yes is class 1
LABEL_MAP = {"No": 0, "Yes": 1}

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path: Path = RAW_PATH, expected_sha256: str = EXPECTED_SHA256) -> pd.DataFrame:
    """Read the original CSV after checking its hash. The file itself is never modified."""
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise ValueError(f"Source hash mismatch for {path}: {actual}")
    # Read everything as text so that type conversion is explicit, not guessed by pandas.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _strip_blanks(column: pd.Series) -> pd.Series:
    """Strip whitespace; empty, whitespace-only and null values all become NaN (object dtype)."""
    text = column.astype("string").str.strip()
    missing = (text.isna() | text.eq("")).to_numpy(dtype=bool)
    return text.astype(object).where(~missing, np.nan)


def parse_numeric(column: pd.Series) -> pd.Series:
    """Blank or whitespace-only -> NaN (permitted missing). Any other non-numeric text raises."""
    text = _strip_blanks(column)
    converted = pd.to_numeric(text, errors="coerce")
    malformed = converted.isna() & text.notna()
    if malformed.any():
        examples = sorted(text[malformed].astype(str).unique())[:5]
        raise ValueError(f"Malformed numeric values in {column.name!r}: {examples}")
    return converted.astype("float64")


def parse_categorical(column: pd.Series) -> pd.Series:
    """Blank -> NaN. Category values are not checked here: unseen ones are handled by the encoder."""
    return _strip_blanks(column)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the six documented feature columns, typed. Fails if any is absent."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    out = pd.DataFrame(index=df.index)
    for name in NUMERIC_FEATURES:
        out[name] = parse_numeric(df[name])
    for name in CATEGORICAL_FEATURES:
        out[name] = parse_categorical(df[name])
    return out


def prepare_target(df: pd.DataFrame) -> pd.Series:
    labels = df[TARGET].str.strip()
    unknown = sorted(set(labels) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"Unexpected {TARGET} labels: {unknown}")
    return labels.map(LABEL_MAP).astype(int).rename(TARGET)


def validate_dataset(df: pd.DataFrame) -> None:
    required = [ID_COLUMN, TARGET, *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate {ID_COLUMN} values")
