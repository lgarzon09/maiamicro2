# esto va primero para que encuentre bien las rutas del proyecto
import Definitions

import pandas as pd
import streamlit as st

from src.ModelController import ModelController

# config de la pagina
st.set_page_config(
    layout="centered", page_title="Clasificador de ODS", page_icon="🌍"
)


# cargo el modelo una sola vez y lo dejo en cache, si no se recarga cada rato
@st.cache_resource
def get_controller():
    return ModelController()


ctrl = get_controller()

# ------- interfaz -------

st.title("Clasificador de textos según los ODS")
st.markdown(
    "Ingresa un **texto libre** y el modelo lo relacionará con uno de los "
    "**Objetivos de Desarrollo Sostenible** de la Agenda 2030. "
    "El texto se procesa con el mismo *pipeline* entrenado en el microproyecto: "
    "`TF-IDF → LSA (TruncatedSVD) → Regresión Logística`."
)

with st.form(key="form_ods"):
    texto = st.text_area(
        "Texto a clasificar",
        height=200,
        placeholder="Escribe o pega aquí el texto que quieres clasificar...",
    )
    submit = st.form_submit_button(label="Clasificar", type="primary")

if submit:
    # si no escribieron nada, aviso y no hago nada mas
    if not ctrl.d_processing.is_valid(texto):
        st.warning("Por favor ingresa un texto antes de clasificar.")
    else:
        ranking = ctrl.predict_ranking(texto, top=5)
        ods, nombre, prob = ranking[0]  # el primero es el mas probable

        col1, col2 = st.columns([1, 2])
        with col1:
            st.caption("🗣 Predicción")
            st.metric("ODS", f"{ods}")
        with col2:
            st.caption("Resultado")
            st.success(f"### ODS {ods} — {nombre}")
            st.metric("Confianza del modelo", f"{prob * 100:.1f}%")

        # muestro tambien los 5 mas probables para que se vea el detalle
        st.caption("Top 5 objetivos más probables")
        tabla = pd.DataFrame(
            {
                "ODS": [f"ODS {o}" for o, _, _ in ranking],
                "Objetivo": [n for _, n, _ in ranking],
                "Probabilidad": [p for _, _, p in ranking],
            }
        )
        st.dataframe(
            tabla,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Probabilidad": st.column_config.ProgressColumn(
                    "Probabilidad", format="%.2f", min_value=0.0, max_value=1.0
                )
            },
        )
        st.bar_chart(tabla.set_index("ODS")["Probabilidad"])

st.caption(
    "Modelo entrenado sobre el OSDG Community Dataset (traducido al español). "
    "Cubre los ODS 1 a 16."
)
