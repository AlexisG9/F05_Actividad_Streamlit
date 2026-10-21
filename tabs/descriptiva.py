import pandas as pd
import streamlit as st


def _numero(df: pd.DataFrame, columna: str) -> pd.Series:
    if columna not in df:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.to_numeric(df[columna], errors="coerce")


def _promedio(df: pd.DataFrame, columna: str) -> str:
    valores = _numero(df, columna).dropna()
    return f"{valores.mean():,.2f}" if not valores.empty else "N/D"


def _formato_promedio(columna: str, valor: float) -> str:
    if pd.isna(valor):
        return "N/D"
    if "costo" in columna.casefold() or "usd" in columna.casefold():
        return f"${valor:,.0f}"
    if "disponibilidad" in columna.casefold() or "%" in columna:
        return f"{valor:,.2f}%"
    return f"{valor:,.2f}"


def mostrar(df: pd.DataFrame) -> None:
    st.header("3. Estadística descriptiva")
    st.markdown(
        "Resume el estado observado de la infraestructura de TI. Las cifras "
        "describen registros y asociaciones; no explican causas ni implican "
        "por sí solas que una configuración sea mejor que otra."
    )

    if df.empty:
        st.warning("No hay registros para resumir.")
        return

    clientes = int(df["Cliente"].nunique()) if "Cliente" in df else len(df)
    media_disponibilidad = _numero(df, "Disponibilidad_%").mean()
    media_incidentes = _numero(df, "Incidentes_Mensuales").mean()
    media_costo = _numero(df, "Costo_TI_Anual_USD").mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes", f"{clientes:,}")
    k2.metric(
        "Disponibilidad promedio",
        f"{media_disponibilidad:.2f}%" if pd.notna(media_disponibilidad) else "N/D",
    )
    k3.metric(
        "Incidentes mensuales promedio",
        f"{media_incidentes:,.1f}" if pd.notna(media_incidentes) else "N/D",
    )
    k4.metric(
        "Costo anual promedio por registro",
        f"${media_costo:,.0f}" if pd.notna(media_costo) else "N/D",
    )
    st.caption(
        "Los promedios se calculan por registro del dataset. Si un cliente "
        "aparece en varias fechas, cuenta más de una vez en esos promedios."
    )

    columnas_numericas = [
        columna
        for columna in df.columns
        if pd.api.types.is_numeric_dtype(df[columna])
    ]

    if columnas_numericas:
        st.subheader("Distribución de una variable numérica")
        variable = st.selectbox(
            "Selecciona una métrica de infraestructura:",
            options=columnas_numericas,
            key="variable_numerica_descriptiva",
        )
        valores = _numero(df, variable).dropna()
        if valores.empty:
            st.info("No hay valores numéricos disponibles para esta variable.")
        else:
            estadisticas = valores.describe().rename_axis("Estadístico").to_frame("Valor")
            st.dataframe(
                estadisticas.style.format({"Valor": "{:,.2f}"}),
                use_container_width=True,
            )
            intervalos = pd.cut(
                valores,
                bins=min(20, max(1, int(valores.nunique()))),
            ).value_counts(sort=False)
            distribucion = pd.DataFrame(
                {
                    "Intervalo": intervalos.index.map(str),
                    "Registros": intervalos.to_numpy(),
                }
            )
            st.caption("Frecuencia por intervalo")
            st.bar_chart(distribucion, x="Intervalo", y="Registros")

    st.subheader("Composición de la infraestructura")
    dimensiones = [
        ("Sector", "Registros por sector"),
        ("Modelo_Operacion", "Registros por modelo de operación"),
        ("Criticidad", "Registros por nivel de criticidad"),
        ("Estado_Proyecto", "Registros por estado del proyecto"),
    ]
    presentes = [(columna, titulo) for columna, titulo in dimensiones if columna in df]
    if presentes:
        columnas = st.columns(min(2, len(presentes)))
        for indice, (columna, titulo) in enumerate(presentes):
            conteo = (
                df[columna]
                .astype("string")
                .fillna("(sin dato)")
                .value_counts()
            )
            with columnas[indice % len(columnas)]:
                st.markdown(f"**{titulo}**")
                distribucion = conteo.rename_axis("Categoría").reset_index(
                    name="Registros"
                )
                st.bar_chart(distribucion, x="Categoría", y="Registros")
    else:
        st.info("No hay columnas categóricas conocidas para mostrar.")

    st.subheader("Comparar segmentos")
    segmentos = [
        columna
        for columna in ["Sector", "Modelo_Operacion", "Criticidad", "Estado_Proyecto"]
        if columna in df
    ]
    if segmentos and columnas_numericas:
        c1, c2 = st.columns(2)
        grupo = c1.selectbox(
            "Agrupar por", segmentos, key="grupo_descriptiva_infraestructura"
        )
        metrica = c2.selectbox(
            "Métrica para comparar",
            columnas_numericas,
            key="metrica_descriptiva_infraestructura",
        )
        temporal = df[[grupo]].copy()
        temporal[metrica] = _numero(df, metrica)
        resumen = (
            temporal.assign(**{grupo: temporal[grupo].astype("string").fillna("(sin dato)")})
            .groupby(grupo, dropna=False)[metrica]
            .agg(Registros="count", Promedio="mean", Mediana="median")
            .reset_index()
        )
        resumen["Promedio"] = resumen["Promedio"].map(
            lambda valor: _formato_promedio(metrica, valor)
        )
        resumen["Mediana"] = resumen["Mediana"].map(
            lambda valor: _formato_promedio(metrica, valor)
        )
        st.dataframe(resumen, use_container_width=True, hide_index=True)

        promedio_por_grupo = (
            temporal.assign(**{grupo: temporal[grupo].astype("string").fillna("(sin dato)")})
            .groupby(grupo, dropna=False)[metrica]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )
        grafica_grupos = promedio_por_grupo.rename_axis(grupo).reset_index(
            name="Promedio"
        )
        st.bar_chart(grafica_grupos, x=grupo, y="Promedio")
        st.caption(
            "El tamaño de cada grupo se muestra junto con el promedio para "
            "dar contexto a la comparación."
        )
    else:
        st.info("Se requieren al menos una dimensión categórica y una métrica numérica.")

    if "Fecha" in df and columnas_numericas:
        fechas = pd.to_datetime(df["Fecha"], errors="coerce")
        mascara_fechas = fechas.notna()
        if mascara_fechas.any():
            st.subheader("Evolución temporal")
            metrica_tiempo = st.selectbox(
                "Métrica para observar por mes",
                columnas_numericas,
                key="metrica_temporal_descriptiva",
            )
            temporal = pd.DataFrame(
                {
                    "Mes": fechas[mascara_fechas].dt.to_period("M").astype(str),
                    "Valor": _numero(df, metrica_tiempo)[mascara_fechas],
                }
            )
            serie_mensual = temporal.groupby("Mes")["Valor"].mean().dropna()
            if not serie_mensual.empty:
                grafica_mensual = serie_mensual.rename_axis("Mes").reset_index(
                    name="Promedio"
                )
                st.line_chart(grafica_mensual, x="Mes", y="Promedio")
                st.caption(
                    "Cada punto es el promedio de los registros con fecha en ese "
                    "mes. La tendencia es descriptiva y no demuestra causalidad."
                )
