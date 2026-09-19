"""Preprocesamiento y catálogo de clases para el clasificador de ODS.

A diferencia de la plantilla original (imágenes de 784 píxeles con escalado y
PCA aplicados por fuera), en este proyecto TODO el preprocesamiento del texto
—limpieza, TF-IDF y reducción de dimensionalidad con LSA— vive DENTRO del
Pipeline de scikit-learn serializado en `model.joblib`. Por eso esta clase no
transforma el texto: se limita a validar la entrada y a traducir el número de
ODS a su nombre oficial.
"""


class DataPreprocessing:

    # Nombres oficiales de los 17 ODS de la Agenda 2030.
    # El dataset del proyecto contiene los ODS 1 a 16 (no hay ODS 17).
    ODS_NOMBRES = {
        1: "Fin de la pobreza",
        2: "Hambre cero",
        3: "Salud y bienestar",
        4: "Educación de calidad",
        5: "Igualdad de género",
        6: "Agua limpia y saneamiento",
        7: "Energía asequible y no contaminante",
        8: "Trabajo decente y crecimiento económico",
        9: "Industria, innovación e infraestructura",
        10: "Reducción de las desigualdades",
        11: "Ciudades y comunidades sostenibles",
        12: "Producción y consumo responsables",
        13: "Acción por el clima",
        14: "Vida submarina",
        15: "Vida de ecosistemas terrestres",
        16: "Paz, justicia e instituciones sólidas",
        17: "Alianzas para lograr los objetivos",
    }

    def __init__(self):
        print("DataPreprocessing.__init__ ->")

    def clean(self, texto):
        """Normalización mínima de la entrada del usuario.

        El pipeline ya realiza minúsculas, quitado de acentos y stopwords;
        aquí solo aseguramos que la entrada sea una cadena no vacía.
        """
        if texto is None:
            return ""
        return str(texto).strip()

    def is_valid(self, texto):
        """Valida que el texto ingresado no esté vacío."""
        return len(self.clean(texto)) > 0

    def get_cat_name(self, ods):
        """Devuelve el nombre del ODS a partir de su número."""
        return self.ODS_NOMBRES.get(int(ods), "ODS desconocido")
