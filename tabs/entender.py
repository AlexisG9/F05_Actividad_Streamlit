import pandas as pd
import streamlit as st


def _tipo_variable(serie: pd.Series, nombre: str = "") -> str:
    """Asigna una etiqueta legible al tipo de variable."""
    if pd.api.types.is_bool_dtype(serie):
        return "Lógica"
    if pd.api.types.is_numeric_dtype(serie):
        return "Numérica"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "Fecha"
    if any(palabra in nombre.casefold() for palabra in ("fecha", "date", "_at")):
        fechas = pd.to_datetime(serie, errors="coerce")
        if serie.notna().sum() and fechas.notna().sum() / serie.notna().sum() >= 0.8:
            return "Fecha"
    return "Texto o categoría"


def _resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    total = len(df)
    for columna in df.columns:
        serie = df[columna]
        no_nulos = int(serie.notna().sum())
        filas.append(
            {
                "Variable": columna,
                "Tipo": _tipo_variable(serie, columna),
                "No nulos": no_nulos,
                "Faltantes": int(serie.isna().sum()),
                "% completo": no_nulos / total * 100 if total else 0,
                "Valores distintos": int(serie.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(filas)


def mostrar(df: pd.DataFrame) -> None:
    """Ayuda a conocer la estructura y el contenido del dataset original."""
    st.header("1. Entender los datos")
    st.markdown(
        "Explora qué representa cada registro, qué variables contiene el "
        "dataset y cómo se distribuyen sus valores. Esta inspección no "
        "transforma ni limpia los datos."
    )

    if df.empty:
        st.warning("No hay registros para explorar.")
        return

    total_celdas = len(df) * len(df.columns)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("Variables", f"{len(df.columns):,}")
    c3.metric("Celdas faltantes", f"{int(df.isna().sum().sum()):,}")
    c4.metric(
        "Completitud",
        f"{100 * (1 - df.isna().sum().sum() / total_celdas):.1f}%"
        if total_celdas
        else "N/D",
    )

    st.subheader("¿Qué contiene el dataset?")
    st.write(
        f"Cada fila representa un registro de información de infraestructura "
        f"de TI. El archivo contiene **{len(df):,} registros** y "
        f"**{len(df.columns):,} variables**. Usa la vista previa para revisar "
        "ejemplos concretos."
    )
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.subheader("Perfil de las variables")
    resumen = _resumen_columnas(df)
    st.dataframe(
        resumen.style.format({"% completo": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )
    faltantes = resumen.loc[resumen["Faltantes"].gt(0)].sort_values(
        "Faltantes", ascending=False
    )
    if faltantes.empty:
        st.success("No se detectaron valores faltantes en el dataset.")
    else:
        st.caption(
            "Las variables con valores faltantes aparecen primero en el "
            "siguiente resumen. Su presencia requiere contexto antes de decidir "
            "cómo tratarlos."
        )
        st.bar_chart(faltantes.set_index("Variable")[["Faltantes"]])

    st.subheader("Composición por tipo de variable")
    conteo_tipos = resumen["Tipo"].value_counts().rename("Variables")
    st.bar_chart(conteo_tipos)

    st.subheader("Explorar una variable")
    variable = st.selectbox(
        "Selecciona una variable para ver su distribución:",
        options=df.columns.tolist(),
        key="variable_entender",
    )
    serie = df[variable]
    tipo = _tipo_variable(serie, variable)
    nulos = int(serie.isna().sum())
    distintos = int(serie.nunique(dropna=True))
    a, b, c = st.columns(3)
    a.metric("Tipo", tipo)
    b.metric("Valores distintos", f"{distintos:,}")
    c.metric("Faltantes", f"{nulos:,}")

    if tipo == "Fecha":
        fechas = pd.to_datetime(serie, errors="coerce").dropna()
        if fechas.empty:
            st.info("No hay fechas válidas para mostrar.")
        else:
            conteo_mensual = (
                fechas.dt.to_period("M")
                .value_counts()
                .sort_index()
                .rename("Registros")
            )
            conteo_mensual.index = conteo_mensual.index.astype(str)
            st.bar_chart(conteo_mensual)
            st.caption(
                f"Periodo observado: {fechas.min():%Y-%m-%d} a "
                f"{fechas.max():%Y-%m-%d}."
            )
    elif tipo == "Numérica":
        estadisticas = serie.describe().rename_axis("Estadístico").to_frame("Valor")
        st.dataframe(estadisticas, use_container_width=True)
        valores = serie.dropna()
        if not valores.empty:
            histograma = pd.cut(valores, bins=min(20, max(1, distintos))).value_counts(
                sort=False
            )
            histograma.index = histograma.index.map(str)
            st.caption("Frecuencia aproximada por intervalo")
            st.bar_chart(histograma.rename("Registros"))
    else:
        frecuencias = (
            serie.astype("string")
            .fillna("(sin dato)")
            .value_counts(dropna=False)
            .head(20)
            .rename_axis("Valor")
            .rename("Registros")
            .reset_index()
        )
        st.bar_chart(frecuencias.set_index("Valor"))
        st.dataframe(frecuencias, use_container_width=True, hide_index=True)
        if distintos > 20:
            st.caption("Se muestran las 20 categorías más frecuentes.")

    st.subheader("Registros duplicados exactos")
    duplicados = int(df.duplicated().sum())
    st.metric("Filas repetidas", f"{duplicados:,}")
    st.caption(
        "Un registro duplicado no siempre es un error; confirma primero si "
        "las filas repetidas representan observaciones distintas en el tiempo."
    )
