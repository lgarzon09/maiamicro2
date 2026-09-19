import Definitions

import os.path as osp

# Librería de serialización usada para persistir el modelo entrenado.
import joblib

from src.DataPreprocessing import DataPreprocessing


class ModelController:
    """Carga el modelo entrenado y expone la lógica de predicción.

    El modelo es un único Pipeline de scikit-learn que integra:
        TfidfVectorizer -> TruncatedSVD (LSA) -> LogisticRegression
    Por eso NO se requieren artefactos separados de escalado ni PCA como en la
    plantilla original: basta con cargar `model.joblib`.
    """

    def __init__(self):
        print("ModelController.__init__ ->")
        # Ruta de la carpeta de modelos y del artefacto entrenado.
        self.models_dir = osp.join(Definitions.ROOT_DIR, "resources/models")
        self.model_path = osp.join(self.models_dir, "model.joblib")

        # Cargar el modelo (Pipeline completo).
        self.model = joblib.load(self.model_path)

        # Clase de preprocesamiento / catálogo de ODS.
        self.d_processing = DataPreprocessing()

    def predict(self, texto):
        """Predice el ODS de un texto libre.

        Retorna una tupla (ods, nombre, probabilidad).
        """
        print("ModelController.predict ->")
        texto = self.d_processing.clean(texto)
        ods = int(self.model.predict([texto])[0])
        prob = float(self.model.predict_proba([texto])[0].max())
        return ods, self.d_processing.get_cat_name(ods), prob

    def predict_ranking(self, texto, top=5):
        """Ranking de los ODS más probables para un texto.

        Retorna una lista de tuplas (ods, nombre, probabilidad) ordenada de
        mayor a menor probabilidad.
        """
        print("ModelController.predict_ranking ->")
        texto = self.d_processing.clean(texto)
        probs = self.model.predict_proba([texto])[0]
        clases = self.model.classes_
        pares = sorted(zip(clases, probs), key=lambda p: p[1], reverse=True)
        ranking = [
            (int(c), self.d_processing.get_cat_name(c), float(p))
            for c, p in pares
        ]
        return ranking[:top]
