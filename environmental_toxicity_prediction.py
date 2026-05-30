"""Utilities for environmental toxicity prediction.

Includes:
- a baseline tabular regression workflow (`train_toxicity_model`)
- a small built-in compound library generator (`build_example_compounds`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ToxicityModelResult:
    mae: float
    r2: float
    n_train: int
    n_test: int


def build_example_compounds() -> pd.DataFrame:
    """Return a small starter set of environmental compounds.

    Notes:
        - `toxicity_index` is an illustrative target to help users run the
          pipeline end-to-end quickly.
        - Replace it with measured endpoints (LC50, EC50, NOEC, etc.) for
          real-world modeling.
    """
    compounds = [
        {
            "compound": "Benzene",
            "family": "VOC",
            "log_kow": 2.13,
            "solubility_mg_l": 1790,
            "molecular_weight": 78.11,
            "pH": 7.0,
            "temperature_c": 20.0,
            "toxicity_index": 62.0,
        },
        {
            "compound": "Toluene",
            "family": "VOC",
            "log_kow": 2.73,
            "solubility_mg_l": 530,
            "molecular_weight": 92.14,
            "pH": 7.0,
            "temperature_c": 20.0,
            "toxicity_index": 58.0,
        },
        {
            "compound": "Atrazine",
            "family": "Pesticide",
            "log_kow": 2.61,
            "solubility_mg_l": 33,
            "molecular_weight": 215.68,
            "pH": 7.0,
            "temperature_c": 20.0,
            "toxicity_index": 71.0,
        },
        {
            "compound": "Phenol",
            "family": "Aromatic",
            "log_kow": 1.46,
            "solubility_mg_l": 84000,
            "molecular_weight": 94.11,
            "pH": 7.0,
            "temperature_c": 20.0,
            "toxicity_index": 49.0,
        },
        {
            "compound": "Bisphenol A",
            "family": "Endocrine disruptor",
            "log_kow": 3.64,
            "solubility_mg_l": 300,
            "molecular_weight": 228.29,
            "pH": 7.0,
            "temperature_c": 20.0,
            "toxicity_index": 76.0,
        },
        {
            "compound": "Chloroform",
            "family": "Halogenated solvent",
            "log_kow": 1.97,
            "solubility_mg_l": 7950,
            "molecular_weight": 119.38,
            "pH": 7.0,
            "temperature_c": 20.0,
            "toxicity_index": 66.0,
        },
    ]
    return pd.DataFrame(compounds)


def train_toxicity_model(
    data: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str] | None = None,
    random_state: int = 42,
) -> ToxicityModelResult:
    """Train and evaluate a baseline model for toxicity prediction."""
    if target_col not in data.columns:
        raise ValueError(f"target_col '{target_col}' not found in data")

    categorical_cols = list(categorical_cols or [])
    feature_cols = [c for c in data.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    x = data[feature_cols]
    y = data[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        [
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=random_state)),
        ]
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    return ToxicityModelResult(
        mae=mean_absolute_error(y_test, preds),
        r2=r2_score(y_test, preds),
        n_train=len(x_train),
        n_test=len(x_test),
    )
