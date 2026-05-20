from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: List[str]


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    csv_path = path if path else str(config.DATA_PATH)
    return pd.read_csv(csv_path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    working_df = df.copy()

    working_df["children"] = working_df["children"].fillna(0)
    working_df["country"] = working_df["country"].fillna("Unknown")

    # NaN in 'agent' means direct booking (no agent), encode as binary flag
    working_df["has_agent"] = working_df["agent"].notnull().astype(int)

    for col in config.LEAKAGE_COLUMNS:
        if col in working_df.columns:
            working_df = working_df.drop(columns=col)

    if config.TARGET_COLUMN not in working_df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in dataset")

    y = working_df[config.TARGET_COLUMN].astype(int)
    X = working_df.drop(columns=[config.TARGET_COLUMN])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def build_data_bundle(path: str | None = None) -> DataBundle:
    df = load_raw_data(path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor(X_train)

    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=X_train.columns.tolist(),
    )
