"""Cositas de apoyo para el clasificador de ODS.

En la plantilla original aca se hacia el preprocesamiento de las imagenes
(escalado, PCA, etc). En mi caso todo eso ya lo hace el pipeline por dentro
(TF-IDF y LSA), asi que esta clase solo la uso para dos cosas: revisar que el
texto no venga vacio y traducir el numero del ODS a su nombre.
"""


class DataPreprocessing:

    # nombres oficiales de los 17 ODS. En los datos solo aparecen del 1 al 16
    # (no hay textos del ODS 17), pero los dejo todos por si acaso.
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
        # el pipeline ya se encarga de minusculas, acentos y stopwords,
        # aca solo quito espacios de sobra y me aseguro que sea texto
        if texto is None:
            return ""
        return str(texto).strip()

    def is_valid(self, texto):
        # sirve para no dejar clasificar cuando la caja esta vacia
        return len(self.clean(texto)) > 0

    def get_cat_name(self, ods):
        # me devuelve el nombre del ODS a partir del numero
        return self.ODS_NOMBRES.get(int(ods), "ODS desconocido")
