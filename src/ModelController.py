import Definitions

import os.path as osp

# joblib es lo que uso para cargar el modelo que guarde en el notebook
import joblib

from src.DataPreprocessing import DataPreprocessing


class ModelController:
    """Se encarga de cargar el modelo y hacer las predicciones.

    Ojo: aca el modelo es un solo Pipeline que ya trae adentro el TF-IDF, el
    LSA (TruncatedSVD) y la regresion logistica. Por eso no necesito cargar el
    scaler ni el pca por separado como en la plantilla original, con el
    model.joblib alcanza.
    """

    def __init__(self):
        print("ModelController.__init__ ->")
        # armo la ruta hasta la carpeta de modelos y el archivo
        self.models_dir = osp.join(Definitions.ROOT_DIR, "resources/models")
        self.model_path = osp.join(self.models_dir, "model.joblib")

        # cargo el pipeline entrenado
        self.model = joblib.load(self.model_path)

        # esta clase me sirve para validar el texto y para los nombres de los ODS
        self.d_processing = DataPreprocessing()

    def predict(self, texto):
        """Devuelve (ods, nombre, probabilidad) para un texto."""
        print("ModelController.predict ->")
        texto = self.d_processing.clean(texto)
        ods = int(self.model.predict([texto])[0])
        prob = float(self.model.predict_proba([texto])[0].max())
        return ods, self.d_processing.get_cat_name(ods), prob

    def predict_ranking(self, texto, top=5):
        """Igual que predict pero devuelve los ODS ordenados de mayor a menor
        probabilidad (una lista de tuplas ods, nombre, probabilidad)."""
        print("ModelController.predict_ranking ->")
        texto = self.d_processing.clean(texto)
        probs = self.model.predict_proba([texto])[0]
        clases = self.model.classes_
        # los ordeno por probabilidad, de mayor a menor
        pares = sorted(zip(clases, probs), key=lambda p: p[1], reverse=True)
        ranking = [
            (int(c), self.d_processing.get_cat_name(c), float(p))
            for c, p in pares
        ]
        return ranking[:top]
