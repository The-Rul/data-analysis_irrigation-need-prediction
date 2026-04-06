from pathlib import Path
import sys

import streamlit as st

st.set_page_config(layout="wide")


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils.data_loader import get_data_paths, load_dataset, summarize_dataset


st.title("Dashboard de Portfolio")
st.caption("Análisis exploratorio orientado a entrevista técnica y decisiones de modelado.")

train_path, test_path = get_data_paths()
df = load_dataset(train_path)
summary = summarize_dataset(df)

st.markdown(
        """
        Este proyecto presenta un caso de predicción de necesidad de riego con una app de Streamlit
        pensada para portfolio. La idea es mostrar criterio de producto, claridad analítica y una
        lectura del dataset que conecte directamente con decisiones de modelado.
        """
    )

metric_cols = st.columns(3)
metric_cols[0].metric("Filas", f"{summary['rows']:,}".replace(",", "."))
metric_cols[1].metric("Variables", summary["feature_count"])
metric_cols[2].metric("Objetivo", summary["target_column"] or "No detectado")

with st.container(border=True):
    st.subheader("Qué puedes explorar")
    st.markdown(
        """
        - `Dashboard de Portfolio`: una vista EDA limpia, explicable y lista para entrevista.
        - Utilidades reutilizables en `streamlit/utils`: carga de datos, resúmenes y gráficos Plotly.
        - Una base preparada para crecer hacia métricas de modelo, predicciones o monitorización.
        """
    )

source_col1, source_col2 = st.columns(2)
with source_col1:
    st.markdown("**Dataset de entrenamiento**")
    st.code(str(train_path.relative_to(Path.cwd())), language="text")

with source_col2:
    st.markdown("**Dataset de prueba**")
    st.code(str(test_path.relative_to(Path.cwd())), language="text")

st.info(
    "La app detecta automáticamente la variable objetivo en el CSV y sigue funcionando aunque "
    "esa columna no esté disponible, lo que hace la exploración más robusta."
)
