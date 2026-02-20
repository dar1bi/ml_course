"""Preprocessing utilities for the bank churn competition dataset."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def select_input_columns(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    drop_cols: Optional[list[str]] = None,
) -> list[str]:
    """Return feature columns used for model training."""
    if drop_cols is None:
        drop_cols = ["id", "CustomerId", "Surname"]

    return [col for col in raw_df.columns if col not in set(drop_cols + [target_col])]


def split_train_val_data(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw data into train and validation subsets."""
    stratify_values = raw_df[target_col] if stratify else None
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )
    return train_df, val_df


def get_feature_types(inputs_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names."""
    numeric_cols = inputs_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = inputs_df.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols


def scale_numeric_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: list[str],
    scaler_numeric: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
    """Scale numeric columns with StandardScaler when scaler_numeric=True."""
    train_processed = train_inputs.copy()
    val_processed = val_inputs.copy()

    if not scaler_numeric or not numeric_cols:
        return train_processed, val_processed, None

    scaler = StandardScaler().fit(train_processed[numeric_cols])
    train_processed.loc[:, numeric_cols] = scaler.transform(train_processed[numeric_cols])
    val_processed.loc[:, numeric_cols] = scaler.transform(val_processed[numeric_cols])

    return train_processed, val_processed, scaler


def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """One-hot encode categorical columns and append encoded features."""
    train_processed = train_inputs.copy()
    val_processed = val_inputs.copy()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_processed[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    train_encoded = pd.DataFrame(
        encoder.transform(train_processed[categorical_cols]),
        columns=encoded_cols,
        index=train_processed.index,
    )
    val_encoded = pd.DataFrame(
        encoder.transform(val_processed[categorical_cols]),
        columns=encoded_cols,
        index=val_processed.index,
    )

    train_processed = pd.concat([train_processed.drop(columns=categorical_cols), train_encoded], axis=1)
    val_processed = pd.concat([val_processed.drop(columns=categorical_cols), val_encoded], axis=1)

    return train_processed, val_processed, encoder


def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    drop_cols: Optional[list[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    scaler_numeric: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    list[str],
    Optional[StandardScaler],
    OneHotEncoder,
]:
    """Preprocess raw train data and return artifacts for model training."""
    input_cols = select_input_columns(raw_df, target_col=target_col, drop_cols=drop_cols)
    train_df, val_df = split_train_val_data(
        raw_df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()
    train_targets = train_df[target_col].astype(int).copy()
    val_targets = val_df[target_col].astype(int).copy()

    numeric_cols, categorical_cols = get_feature_types(train_inputs)

    train_inputs, val_inputs, scaler = scale_numeric_features(
        train_inputs,
        val_inputs,
        numeric_cols=numeric_cols,
        scaler_numeric=scaler_numeric,
    )

    train_inputs, val_inputs, encoder = encode_categorical_features(
        train_inputs,
        val_inputs,
        categorical_cols=categorical_cols,
    )

    return (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        input_cols,
        scaler,
        encoder,
    )


def preprocess_new_data(
    raw_df: pd.DataFrame,
    input_cols: list[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder,
    model_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Preprocess new data (e.g., test.csv) with fitted scaler and encoder."""
    data = raw_df[input_cols].copy()

    categorical_cols = list(encoder.feature_names_in_)
    numeric_cols = [col for col in input_cols if col not in categorical_cols]

    if scaler is not None and numeric_cols:
        scale_cols = [col for col in numeric_cols if col in set(getattr(scaler, "feature_names_in_", numeric_cols))]
        data.loc[:, scale_cols] = scaler.transform(data[scale_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    encoded = pd.DataFrame(
        encoder.transform(data[categorical_cols]),
        columns=encoded_cols,
        index=data.index,
    )

    data = pd.concat([data.drop(columns=categorical_cols), encoded], axis=1)

    if model_features is not None:
        data = data.reindex(columns=model_features, fill_value=0.0)

    return data
