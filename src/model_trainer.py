from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

from src import config
from src.data_loader import DataBundle

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class KerasClassifier(BaseEstimator, ClassifierMixin):
    """Adaptador compatible con scikit-learn para un clasificador binario en Keras.

    :param epochs: Número de épocas de entrenamiento.
    :param batch_size: Tamaño de lote usado por Keras durante el ajuste.
    :param learning_rate: Tasa de aprendizaje del optimizador Adam.
    """

    def __init__(self, epochs: int = 50, batch_size: int = 256, learning_rate: float = 0.001) -> None:
        """Guarda los hiperparámetros de entrenamiento de la red neuronal.

        :param epochs: Número de pasadas completas sobre el conjunto de datos.
        :param batch_size: Número de muestras procesadas en cada actualización.
        :param learning_rate: Tasa de aprendizaje del optimizador.
        """

        # Número de veces que el modelo verá todos los datos de entrenamiento
        self.epochs = epochs

        # Número de muestras que se procesan juntas antes de actualizar los pesos
        self.batch_size = batch_size

        # Velocidad a la que el optimizador ajusta los pesos
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Ajusta la red neuronal sobre matrices de características ya transformadas.

        :param X: Matriz de características preprocesadas.
        :param y: Etiquetas binarias del entrenamiento.
        :returns: La propia instancia ajustada.
        """

        # Keras trabaja con float32; el preprocesador de sklearn devuelve float64
        X = np.array(X, dtype="float32")
        y = np.array(y, dtype="float32")

        # Arquitectura de la red: tres capas densas (fully connected)
        #   - Capa 1: 64 neuronas con ReLU
        #   - Capa 2: 32 neuronas con ReLU
        #   - Capa 3: 1 neurona con sigmoide → devuelve una probabilidad entre 0 y 1
        self.model_ = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])

        # Configuración del entrenamiento:
        #   - Adam: optimizador que adapta el learning rate automáticamente
        #   - binary_crossentropy: función de pérdida estándar para clasificación binaria
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
        )

        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Devuelve probabilidades en el formato de dos columnas de scikit-learn.

        :param X: Matriz de entrada ya preprocesada.
        :returns: Matriz ``(n_muestras, 2)`` con probabilidades por clase.
        """
        X = np.array(X, dtype="float32")

        # model_.predict devuelve shape (n, 1) → flatten lo convierte a (n,)
        proba = self.model_.predict(X, verbose=0).flatten()

        # sklearn espera columnas [prob_clase_0, prob_clase_1]
        return np.column_stack([1 - proba, proba])

    def predict(self, X) -> np.ndarray:
        """Predice la clase binaria usando un umbral de probabilidad de 0.5.

        :param X: Matriz de entrada ya preprocesada.
        :returns: Vector de predicciones binarias.
        """
        
        # Si la probabilidad de cancelación supera 0.5 → predice 1 (cancelado)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CatBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Adaptador de CatBoost para trabajar con ``DataFrame`` sin preprocesado común.

    A diferencia del resto de estimadores del proyecto, este wrapper no usa el
    :class:`~sklearn.compose.ColumnTransformer` compartido. Espera datos
    tabulares en bruto y normaliza las columnas categóricas antes de delegar en
    CatBoost.

    :param iterations: Número máximo de iteraciones del algoritmo.
    :param learning_rate: Tasa de aprendizaje usada por CatBoost.
    :param depth: Profundidad máxima de los árboles.
    """

    def __init__(self, iterations: int = 100, learning_rate: float = 0.1, depth: int = 6) -> None:
        """Guarda los hiperparámetros de CatBoost usados durante el ajuste.

        :param iterations: Número máximo de iteraciones del algoritmo.
        :param learning_rate: Tasa de aprendizaje usada por CatBoost.
        :param depth: Profundidad máxima de los árboles.
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth

    def _prepare_features(self, X):
        if not hasattr(X, "copy"):
            return X

        X_prepared = X.copy()
        if not hasattr(X_prepared, "select_dtypes"):
            return X_prepared

        cat_cols = X_prepared.select_dtypes(exclude=["number"]).columns.tolist()
        for col in cat_cols:
            X_prepared[col] = X_prepared[col].fillna("Unknown").astype(str)
        return X_prepared

    def fit(self, X, y):
        """Ajusta el clasificador CatBoost envuelto sobre variables en bruto.

        :param X: Datos tabulares sin transformar.
        :param y: Etiquetas binarias del entrenamiento.
        :returns: La propia instancia ajustada.
        """
        X_prepared = self._prepare_features(X)
        cat_features = None
        if hasattr(X_prepared, "select_dtypes"):
            cat_features = X_prepared.select_dtypes(exclude=["number"]).columns.tolist()

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            verbose=0,
        )
        self.model_.fit(X_prepared, y, cat_features=cat_features)
        return self

    def evaluate(self, X_test, y_test) -> None:
        """Imprime métricas de diagnóstico para inspección manual de CatBoost.

        Este helper no se utiliza en la canalización principal de entrenamiento.

        :param X_test: Variables predictoras del conjunto de prueba.
        :param y_test: Etiquetas reales del conjunto de prueba.
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        conf = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred)

        print(f"Accuracy Score of CatBoost Classifier is : {acc}")
        print(f"Confusion Matrix : \n{conf}")
        print(f"Classification Report : \n{clf_report}")

    def predict_proba(self, X) -> np.ndarray:
        """Devuelve las probabilidades de clase estimadas por CatBoost.

        :param X: Muestras sobre las que se desea inferir.
        :returns: Matriz de probabilidades por clase.
        """
        X_prepared = self._prepare_features(X)
        return self.model_.predict_proba(X_prepared)

    def predict(self, X) -> np.ndarray:
        """Predice la variable objetivo binaria con el modelo CatBoost envuelto.

        :param X: Muestras sobre las que se desea inferir.
        :returns: Vector de predicciones binarias.
        """
        X_prepared = self._prepare_features(X)
        return self.model_.predict(X_prepared).flatten()


@dataclass
class TrainedModels:
    """Agrupa los modelos ajustados y las rutas de sus artefactos persistidos.

    :param models: Diccionario ``nombre -> modelo`` ya entrenado.
    :param model_paths: Diccionario ``nombre -> ruta`` del artefacto serializado.
    """
    models: Dict[str, BaseEstimator]
    model_paths: Dict[str, Path]


class ModelTrainer:
    """Entrena, persiste y expone los modelos candidatos configurados.

    :param data_bundle: Paquete de datos preparado para entrenamiento.
    :param quick_mode: Si es ``True``, reduce el coste computacional del ajuste.
    :param skip_neural_net: Si es ``True``, omite el modelo neuronal.
    """

    def __init__(self, data_bundle: DataBundle, quick_mode: bool = False, skip_neural_net: bool = False) -> None:
        """Almacena los datos y las opciones de ejecución del entrenamiento.

        :param data_bundle: Datos particionados y preprocesador compartido.
        :param quick_mode: Activa una versión más rápida del entrenamiento.
        :param skip_neural_net: Omite el modelo neuronal cuando vale ``True``.
        """
        self.data = data_bundle
        self.quick_mode = quick_mode
        self.skip_neural_net = skip_neural_net

    def _get_estimators(self) -> Dict[str, ClassifierMixin]:
        estimators: Dict[str, ClassifierMixin] = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=config.RANDOM_STATE),
            "random_forest": RandomForestClassifier(
                n_estimators=50 if self.quick_mode else 200,
                max_depth=None,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            "gradient_boosting": self._build_gradient_boosting(),
        }

        if HAS_CATBOOST:
            estimators["catboost"] = CatBoostClassifierWrapper(
                iterations=50 if self.quick_mode else 100,
            )

        if not self.skip_neural_net:
            if not HAS_KERAS:
                raise ImportError("Instala TensorFlow con: pip install tensorflow")
            estimators["neural_network"] = KerasClassifier(
                epochs=10 if self.quick_mode else 50,
            )

        return estimators

    def _build_gradient_boosting(self) -> ClassifierMixin:
        if HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=40 if self.quick_mode else 200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )

        # Fallback when XGBoost is not available.
        return GradientBoostingClassifier(random_state=config.RANDOM_STATE)

    def train_all(self) -> TrainedModels:
        """Entrena todos los modelos configurados y guarda sus artefactos individuales.

        Este método persiste un artefacto ``models\\<name>.pkl`` por candidato.
        La selección del ganador y su almacenamiento en ``best_model.pkl`` se
        delegan en :meth:`save_best_model`.

        :returns: Instancia de :class:`TrainedModels` con los modelos ajustados y
            sus rutas de salida.
        :raises ImportError: Si se solicita la red neuronal y TensorFlow no está
            instalado.
        """
        estimators = self._get_estimators()
        trained: Dict[str, BaseEstimator] = {}
        paths: Dict[str, Path] = {}

        for name, estimator in estimators.items():
            if name == "catboost":
                model = estimator
            else:
                model = Pipeline(
                    steps=[
                        ("preprocessor", self.data.preprocessor),
                        ("classifier", estimator),
                    ]
                )
            model.fit(self.data.X_train, self.data.y_train)
            trained[name] = model

            model_path = config.MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, model_path)
            paths[name] = model_path

        return TrainedModels(models=trained, model_paths=paths)

    @staticmethod
    def save_best_model(model: BaseEstimator, model_name: str) -> Path:
        """Guarda el modelo ganador como un artefacto estructurado ``best_model.pkl``.

        El objeto serializado es un diccionario con las claves ``model_name`` y
        ``model``, por lo que debe cargarse mediante
        :func:`src.predictor.load_best_model`.

        :param model: Modelo entrenado seleccionado como ganador.
        :param model_name: Nombre lógico del modelo ganador.
        :returns: Ruta del artefacto generado.
        """
        artifact = {
            "model_name": model_name,
            "model": model,
        }
        out_path = config.MODELS_DIR / "best_model.pkl"
        joblib.dump(artifact, out_path)
        return out_path


def ensure_probabilities(model: BaseEstimator, X) -> np.ndarray:
    """Obtiene puntuaciones de la clase positiva usando la mejor interfaz disponible.

    El helper prioriza ``predict_proba()``, después ``decision_function()``
    normalizada y, como último recurso, usa las predicciones directas del
    modelo.

    :param model: Estimador entrenado compatible con alguna interfaz de scoring.
    :param X: Muestras sobre las que se calcula la puntuación.
    :returns: Vector unidimensional con puntuaciones de la clase positiva.
    """
    if hasattr(model, "predict_proba"):
        # devuelve la probabilidad de la clase positiva (índice 1)(la de cancelacion) para cada muestra
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    return model.predict(X)
