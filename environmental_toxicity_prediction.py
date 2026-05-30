"""Utilities and report generation for environmental toxicity prediction.

The module is intentionally usable in two modes:
- report/compound inspection mode with only the Python standard library
- model-training mode when pandas and scikit-learn are installed
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from statistics import mean
from typing import Any, Iterable


EXAMPLE_COMPOUNDS: list[dict[str, str | float]] = [
    {
        "compound": "Benzene",
        "family": "VOC",
        "log_kow": 2.13,
        "solubility_mg_l": 1790.0,
        "molecular_weight": 78.11,
        "pH": 7.0,
        "temperature_c": 20.0,
        "toxicity_index": 62.0,
    },
    {
        "compound": "Toluene",
        "family": "VOC",
        "log_kow": 2.73,
        "solubility_mg_l": 530.0,
        "molecular_weight": 92.14,
        "pH": 7.0,
        "temperature_c": 20.0,
        "toxicity_index": 58.0,
    },
    {
        "compound": "Atrazine",
        "family": "Pesticide",
        "log_kow": 2.61,
        "solubility_mg_l": 33.0,
        "molecular_weight": 215.68,
        "pH": 7.0,
        "temperature_c": 20.0,
        "toxicity_index": 71.0,
    },
    {
        "compound": "Phenol",
        "family": "Aromatic",
        "log_kow": 1.46,
        "solubility_mg_l": 84000.0,
        "molecular_weight": 94.11,
        "pH": 7.0,
        "temperature_c": 20.0,
        "toxicity_index": 49.0,
    },
    {
        "compound": "Bisphenol A",
        "family": "Endocrine disruptor",
        "log_kow": 3.64,
        "solubility_mg_l": 300.0,
        "molecular_weight": 228.29,
        "pH": 7.0,
        "temperature_c": 20.0,
        "toxicity_index": 76.0,
    },
    {
        "compound": "Chloroform",
        "family": "Halogenated solvent",
        "log_kow": 1.97,
        "solubility_mg_l": 7950.0,
        "molecular_weight": 119.38,
        "pH": 7.0,
        "temperature_c": 20.0,
        "toxicity_index": 66.0,
    },
]


@dataclass
class ToxicityModelResult:
    mae: float
    r2: float
    n_train: int
    n_test: int


def _require_dependency(module_name: str, install_name: str | None = None) -> None:
    if find_spec(module_name) is None:
        package = install_name or module_name
        raise ImportError(
            f"Missing optional dependency '{package}'. Install it to use model "
            f"training, for example: python -m pip install {package}"
        )


def build_example_compounds(as_dataframe: bool = True) -> Any:
    """Return a starter set of environmental compounds.

    Args:
        as_dataframe: When True, return a pandas DataFrame. When False, return a
            list of dictionaries that works without third-party dependencies.

    Notes:
        `toxicity_index` is an illustrative target to help users run the
        workflow end-to-end quickly. Replace it with measured LC50, EC50, NOEC,
        or similar endpoints for real-world modeling.
    """
    compounds = [compound.copy() for compound in EXAMPLE_COMPOUNDS]
    if not as_dataframe:
        return compounds

    _require_dependency("pandas")
    import pandas as pd

    return pd.DataFrame(compounds)


def generate_compound_report(
    compounds: Iterable[dict[str, str | float]] | None = None,
) -> str:
    """Build a Markdown report for the example compound set."""
    rows = list(compounds or build_example_compounds(as_dataframe=False))
    toxicity_values = [float(row["toxicity_index"]) for row in rows]
    highest = max(rows, key=lambda row: float(row["toxicity_index"]))
    lowest = min(rows, key=lambda row: float(row["toxicity_index"]))

    report = [
        "# Environmental Toxicity Prediction Report",
        "",
        "## Scope",
        "This report summarizes the built-in starter compound set for the "
        "environmental toxicity prediction workflow.",
        "",
        "## Compound Dataset",
        "| Compound | Family | logKow | Solubility (mg/L) | Molecular Weight | pH | Temperature (°C) | Toxicity Index |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        report.append(
            "| {compound} | {family} | {log_kow:.2f} | {solubility_mg_l:.1f} | "
            "{molecular_weight:.2f} | {pH:.1f} | {temperature_c:.1f} | "
            "{toxicity_index:.1f} |".format(**row)
        )

    report.extend(
        [
            "",
            "## Summary Metrics",
            f"- Number of compounds: {len(rows)}",
            f"- Average illustrative toxicity index: {mean(toxicity_values):.2f}",
            f"- Lowest illustrative toxicity index: {lowest['compound']} ({lowest['toxicity_index']:.1f})",
            f"- Highest illustrative toxicity index: {highest['compound']} ({highest['toxicity_index']:.1f})",
            "",
            "## Modeling Notes",
            "- The toxicity index values are illustrative placeholders for demo use.",
            "- Replace `toxicity_index` with measured LC50, EC50, NOEC, or a similar endpoint before making scientific decisions.",
            "- The training pipeline accepts the generated compounds directly when pandas and scikit-learn are installed.",
        ]
    )
    return "\n".join(report) + "\n"


def train_toxicity_model(
    data: Any,
    target_col: str,
    categorical_cols: Iterable[str] | None = None,
    random_state: int = 42,
) -> ToxicityModelResult:
    """Train and evaluate a baseline model for toxicity prediction."""
    _require_dependency("sklearn", "scikit-learn")

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

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


if __name__ == "__main__":
    print(generate_compound_report())
