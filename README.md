# Clasificador de textos según los ODS 🌍

Aplicación web (Streamlit) del **Microproyecto 2 — Machine Learning no supervisado (MAIA)**.
Recibe un **texto libre** y lo relaciona con uno de los **Objetivos de Desarrollo
Sostenible (ODS)** de la Agenda 2030, usando el mismo *pipeline* entrenado en el
notebook del proyecto:

```
TF-IDF  →  LSA (TruncatedSVD, 100 componentes)  →  Regresión Logística (C=10)
```

El proyecto sigue la **estructura de la plantilla del curso**, adaptada a un
problema de clasificación de texto (un único artefacto `model.joblib`, ya que el
Pipeline integra el preprocesamiento completo; no se requieren `scaler`/`pca`
por separado).

## Estructura

```
ODS_Classifier/
├── Definitions.py                  # ROOT_DIR y rutas del proyecto
├── configuration.conf
├── requirements.txt
├── streamlit_app.py                # Interfaz (archivo principal)
├── resources/
│   ├── batch/streamlit.bat         # Lanzador local (Windows)
│   └── models/model.joblib         # Pipeline entrenado (un solo artefacto)
└── src/
    ├── ModelController.py          # Carga del modelo y predicción
    └── DataPreprocessing.py        # Validación de entrada y catálogo de ODS
```

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Despliegue en Streamlit Community Cloud

1. Sube este proyecto a un repositorio de **GitHub público**.
2. En [share.streamlit.io](https://share.streamlit.io) → **Create app** →
   *Deploy a public app from GitHub*.
3. **Repository:** tu repo · **Main file path:** `ODS_Classifier/streamlit_app.py`
   (o `streamlit_app.py` si subes el contenido de esta carpeta a la raíz del repo).
4. **Advanced settings → Python 3.12** (⚠️ no 3.10: `scikit-learn 1.9` / `numpy 2.5`
   requieren Python 3.11+).
5. **Deploy** y copia la URL pública.

> ⚠️ Las versiones de `requirements.txt` están fijadas a las mismas con las que se
> entrenó el modelo, para garantizar que `model.joblib` se cargue sin errores.
