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
    # La columna "children" se rellena con 0 donde hay valores nulos. (Significa que no hay niños en la reserva)
    working_df["children"] = working_df["children"].fillna(0)
    
    # La columna "country" se rellena con "Unknown" donde hay valores nulos. (Significa que el país de origen no se especificó)
    working_df["country"] = working_df["country"].fillna("Unknown")
    # Para evitar la alta cardinalidad en la columna "country", se agrupan los países menos frecuentes en una categoría "Other". Se identifican los 10 países más comunes y se reemplazan los demás con "Other". Esto ayuda a reducir la cantidad de categorías únicas y mejora la capacidad del modelo para generalizar.
    top_countries = working_df["country"].value_counts().nlargest(10).index
    working_df["country"] = working_df["country"].where(
        working_df["country"].isin(top_countries),
        "Other",
    )
    
    # Agente nulo significa reserva directa (sin agente), asignar un indicador binario, astype(int) convierte True a 1 y False a 0
    # Se añade la columna "has_agent", que vale 1 si la columna "agent" no es nula (es decir, la reserva fue hecha por un agente) y 0 si es nula (reserva directa).
    working_df["has_agent"] = working_df["agent"].notnull().astype(int)
    
    # Se eliminan columnas de fuga de información, como "reservation_status" y "reservation_status_date", que podrían revelar el resultado de la reserva (si se canceló o no) y la fecha de esa información, lo que no estaría disponible en el momento de la predicción.
    # La idea es predecir la cancelación de la reserva antes de que ocurra, por lo que no se deben incluir columnas que contengan información que solo estaría disponible después de la reserva.
    for col in config.LEAKAGE_COLUMNS:
        if col in working_df.columns:
            working_df = working_df.drop(columns=col)
    # Comprobamos que la columna objetivo esta en el DataFrame, por seguridad. Si no está, se lanza un error para evitar problemas posteriores en el pipeline de entrenamiento.
    if config.TARGET_COLUMN not in working_df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in dataset")
    # Separamos la columna objetivo del resto de las características. La columna objetivo se convierte a tipo entero (0 o 1) para asegurar que el modelo de clasificación pueda procesarla correctamente.
    y = working_df[config.TARGET_COLUMN].astype(int)
    # Se eliminan del DataFrame las columnas objetivo, dejando solo las variables predictoras (X) para el entrenamiento del modelo. Esto asegura que el modelo no tenga acceso a la información de la variable objetivo durante el proceso de aprendizaje.
    X = working_df.drop(columns=[config.TARGET_COLUMN])
    
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Se identifican las columnas numéricas y categóricas en el DataFrame X. Las columnas numéricas se seleccionan utilizando select_dtypes con include=["number"].
    # mientras que las columnas categóricas se seleccionan utilizando exclude=["number"].
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Numéricas: median + StandardScaler 
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Categóricas: most_frequent + OneHotEncoder (handle_unknown="ignore" para evitar errores con categorías no vistas en el conjunto de entrenamiento, sparse_output=False para obtener una matriz densa)
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

# La función split_data utiliza train_test_split de scikit-learn para dividir el conjunto de datos en entrenamiento y prueba. Se especifica un tamaño de prueba (test_size) y una semilla aleatoria (random_state) para garantizar la reproducibilidad.
#  Además, se utiliza stratify=y para asegurar que la proporción de clases en la variable objetivo se mantenga igual en ambos conjuntos, lo que es especialmente importante en problemas de clasificación con clases desbalanceadas.
def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

# La función build_data_bundle es la función principal que orquesta todo el proceso de carga y preparación de datos. Primero, carga los datos crudos utilizando load_raw_data, luego prepara las características y la variable objetivo con prepare_features, divide los datos en conjuntos de entrenamiento y prueba con split_data, y finalmente construye el preprocesador con build_preprocessor.
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
