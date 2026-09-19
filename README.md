# Clasificador de textos por ODS

App en Streamlit para el Microproyecto 2 de Machine Learning no supervisado (MAIA).

La idea es sencilla: uno escribe un texto y la app dice a cuál de los Objetivos
de Desarrollo Sostenible (ODS) de la Agenda 2030 se parece más. Por dentro usa el
mismo modelo que entrené en el notebook:

```
TF-IDF  ->  LSA (TruncatedSVD, 100 componentes)  ->  Regresión Logística (C=10)
```

Seguí la estructura de la plantilla del curso, pero adaptada al problema de texto.
Como el pipeline ya hace todo el preprocesamiento por dentro, me quedó un solo
`model.joblib` (no necesité guardar el scaler ni el pca aparte). El modelo está
entrenado con el OSDG Community Dataset (traducido al español) y clasifica los
ODS del 1 al 16.

## Cómo la corro en mi máquina

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Cómo la subí a Streamlit Cloud

1. Subí el proyecto a un repo público de GitHub.
2. En [share.streamlit.io](https://share.streamlit.io) → **Create app** →
   *Deploy a public app from GitHub*.
3. Elegí el repo, branch `main` y como archivo principal `streamlit_app.py`.
4. En **Advanced settings** puse **Python 3.12** (con 3.10 no sirve porque
   `scikit-learn 1.9` y `numpy 2.5` piden 3.11 o superior).
5. Deploy.

## Estructura

```
ODS_Classifier/
├── Definitions.py              # rutas del proyecto
├── configuration.conf
├── requirements.txt
├── streamlit_app.py            # la app (archivo principal)
├── resources/
│   ├── batch/streamlit.bat     # para lanzarla en local (Windows)
│   └── models/model.joblib     # el modelo entrenado
└── src/
    ├── ModelController.py      # carga el modelo y predice
    └── DataPreprocessing.py    # valida el texto y guarda los nombres de los ODS
```

> Nota: dejé fijas las versiones en `requirements.txt` (las mismas con las que
> entrené) para que el `model.joblib` cargue sin problemas en el despliegue.
