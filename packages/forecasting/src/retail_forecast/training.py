import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = (
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_mean_4",
)


def train_random_forest(
    features: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end_exclusive: pd.Timestamp,
) -> RandomForestRegressor:
    """Fit a first candidate using only eligible training-period rows."""
    if train_end_exclusive <= train_start:
        raise ValueError("Training end must be after training start.")

    training = features.loc[
        features["week_start"].ge(train_start) & features["week_start"].lt(train_end_exclusive)
    ].dropna(subset=["units_sold", *FEATURE_COLUMNS])

    if training.empty:
        raise ValueError("No usable training rows.")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        training[list(FEATURE_COLUMNS)].astype("float64"),
        training["units_sold"].astype("float64"),
    )

    return model


def train_random_forest_with_product_id(
    features: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end_exclusive: pd.Timestamp,
) -> Pipeline:
    """Fit the same RF settings with product identity added."""
    if train_end_exclusive <= train_start:
        raise ValueError("Training end must be after training start.")

    training = features.loc[
        features["week_start"].ge(train_start) & features["week_start"].lt(train_end_exclusive)
    ].dropna(subset=["StockCode", "units_sold", *FEATURE_COLUMNS])

    if training.empty:
        raise ValueError("No usable training rows.")

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "product",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["StockCode"],
            ),
            ("sales_history", "passthrough", list(FEATURE_COLUMNS)),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(
        training[["StockCode", *FEATURE_COLUMNS]],
        training["units_sold"].astype("float64"),
    )

    return pipeline
