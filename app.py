from pathlib import Path

import pandas as pd
import streamlit as st

from tabs.entender import mostrar as mostrar_entender
from tabs.limpiar import mostrar as mostrar_limpiar
from tabs.descriptiva import mostrar as mostrar_descriptiva
from tabs.predictiva import mostrar as mostrar_predictiva
from tabs.prescriptiva import mostrar as mostrar_prescriptiva
from tabs.decisiones import mostrar as mostrar_decisiones


RUTA_DATASET = (
    Path(__file__).resolve().parent
    / "data"
    / "Dataset_Infraestructura_TI_Clientes.csv"
)


st.set_page_config(
    page_title="IT Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 IT Analytics")
st.markdown(
    "Aplicación didáctica para analizar clientes y convertir evidencia "
    "en decisiones de infraestructura de TI."
)


@st.cache_data
def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """Carga el CSV original sin modificar ni imputar sus datos."""
    return pd.read_csv(ruta)


if not RUTA_DATASET.is_file():
    st.error(
        "No se encontró el dataset esperado: "
        f"`{RUTA_DATASET.relative_to(Path(__file__).resolve().parent)}`. "
        "Verifica que el archivo esté dentro de la carpeta `data`."
    )
    st.stop()

try:
    df = cargar_dataset(RUTA_DATASET)
except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"No se pudo leer el dataset: {error}")
    st.stop()

if df.empty:
    st.error("El dataset está vacío. Revisa el archivo CSV antes de continuar.")
    st.stop()


etiquetas_pestanas = [
    "1. Entender",
    "2. Limpiar",
    "3. Descriptiva",
    "4. Predictiva",
    "5. Prescriptiva",
    "6. Decisiones",
]
tab_entender, tab_limpiar, tab_descriptiva, tab_predictiva, tab_prescriptiva, tab_decisiones = st.tabs(
    etiquetas_pestanas
)

with tab_entender:
    mostrar_entender(df)

with tab_limpiar:
    mostrar_limpiar(df)

df_analisis = st.session_state.get("df_limpio", df)

with tab_descriptiva:
    mostrar_descriptiva(df_analisis)

with tab_predictiva:
    mostrar_predictiva(df_analisis)

with tab_prescriptiva:
    mostrar_prescriptiva(df_analisis)

with tab_decisiones:
    mostrar_decisiones(df_analisis)