import pandas as pd
import streamlit as st


def _reporte_calidad(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for columna in df.columns:
        serie = df[columna]
        filas.append(
            {
                "Variable": columna,
                "Tipo": str(serie.dtype),
                "Faltantes": int(serie.isna().sum()),
                "% faltante": 100 * serie.isna().mean() if len(df) else 0,
                "Valores únicos": int(serie.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(filas)


def _hallazgos_rango(df: pd.DataFrame) -> pd.DataFrame:
    """Señala valores sospechosos con reglas generales, sin corregirlos."""
    hallazgos = []
    for columna in df.columns:
        if not pd.api.types.is_numeric_dtype(df[columna]):
            continue
        valores = pd.to_numeric(df[columna], errors="coerce")
        negativos = int(valores.lt(0).sum())
        if negativos:
            hallazgos.append(
                {
                    "Variable": columna,
                    "Regla de revisión": "Valor menor que 0",
                    "Registros": negativos,
                }
            )
        nombre = columna.casefold()
        if "%" in columna or "porcentaje" in nombre:
            fuera = int((valores.lt(0) | valores.gt(100)).sum())
            if fuera:
                hallazgos.append(
                    {
                        "Variable": columna,
                        "Regla de revisión": "Porcentaje fuera de 0 a 100",
                        "Registros": fuera,
                    }
                )
    return pd.DataFrame(
        hallazgos, columns=["Variable", "Regla de revisión", "Registros"]
    )


def _resumen_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    duplicados = df.duplicated(keep=False)
    if not duplicados.any():
        return df.head(0)
    return df.loc[duplicados].head(20)


def mostrar(df: pd.DataFrame) -> None:
    st.header("2. Limpiar los datos")
    st.markdown(
        "Revisa la calidad del dataset de infraestructura y aplica solo los "
        "tratamientos que tengan sentido para el ejercicio. El archivo de "
        "origen no se modifica; las decisiones se aplican a una copia de trabajo."
    )

    if df.empty:
        st.warning("No hay registros para limpiar.")
        return

    if "df_limpio" not in st.session_state:
        st.session_state.df_limpio = df.copy()
        st.session_state.limpieza_acciones = [
            "Se creó una copia de trabajo sin cambios."
        ]

    st.subheader("Diagnóstico de calidad")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros de origen", f"{len(df):,}")
    c2.metric("Variables", f"{len(df.columns):,}")
    c3.metric("Celdas faltantes", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Filas duplicadas exactas", f"{int(df.duplicated().sum()):,}")

    perfil = _reporte_calidad(df)
    st.dataframe(
        perfil.style.format({"% faltante": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Revisar valores que podrían requerir validación"):
        hallazgos = _hallazgos_rango(df)
        if hallazgos.empty:
            st.success("No se encontraron valores negativos ni porcentajes fuera de rango.")
        else:
            st.dataframe(hallazgos, use_container_width=True, hide_index=True)
        st.caption(
            "Son reglas generales de revisión, no errores confirmados. "
            "Los valores extremos y los límites válidos dependen del contexto "
            "de cada variable."
        )

    duplicados = _resumen_duplicados(df)
    if not duplicados.empty:
        with st.expander("Ver muestra de filas duplicadas exactas"):
            st.dataframe(duplicados, use_container_width=True, hide_index=True)
            st.caption(
                "Si una fila representa una medición en otro momento, puede "
                "ser válida aunque coincida con otra."
            )

    st.subheader("Aplicar decisiones a la copia de trabajo")
    columnas_texto = df.select_dtypes(include=["object", "string"]).columns.tolist()
    columnas_con_faltantes = [
        columna for columna in df.columns if df[columna].isna().any()
    ]

    with st.form("formulario_limpieza"):
        quitar_duplicados = st.checkbox(
            "Quitar filas completamente duplicadas",
            value=False,
            key="quitar_duplicados",
        )
        quitar_espacios = st.checkbox(
            "Eliminar espacios al inicio y al final de los valores de texto",
            value=False,
            disabled=not columnas_texto,
            key="quitar_espacios_texto",
        )

        if columnas_con_faltantes:
            variable_faltantes = st.selectbox(
                "Variable con faltantes para tratar (opcional)",
                options=["No tratar faltantes"] + columnas_con_faltantes,
                key="variable_faltantes_limpieza",
            )
        else:
            variable_faltantes = "No tratar faltantes"
            st.caption("El dataset no contiene valores faltantes.")

        estrategia = "Conservar faltantes"
        if variable_faltantes != "No tratar faltantes":
            es_numerica = pd.api.types.is_numeric_dtype(df[variable_faltantes])
            estrategias = ["Conservar faltantes", "Eliminar filas con faltantes"]
            if es_numerica and df[variable_faltantes].notna().any():
                estrategias.append("Completar con la mediana")
            elif df[variable_faltantes].notna().any():
                estrategias.append("Completar con el valor más frecuente")
            estrategia = st.selectbox(
                "Tratamiento para esa variable",
                options=estrategias,
                key="estrategia_faltantes_limpieza",
            )

        aplicar = st.form_submit_button("Aplicar limpieza seleccionada")

    if aplicar:
        limpio = df.copy()
        acciones = []

        if quitar_duplicados:
            antes = len(limpio)
            limpio = limpio.drop_duplicates().copy()
            acciones.append(
                f"Se quitaron {antes - len(limpio):,} filas duplicadas exactas."
            )

        if quitar_espacios:
            celdas_ajustadas = 0
            for columna in columnas_texto:
                valores = limpio[columna]
                texto = valores.astype("string")
                recortado = texto.str.strip()
                celdas_ajustadas += int((texto.notna() & texto.ne(recortado)).sum())
                limpio[columna] = recortado.where(texto.notna(), pd.NA)
            acciones.append(
                f"Se recortaron espacios al inicio o final en {celdas_ajustadas:,} celdas de texto."
            )

        if variable_faltantes != "No tratar faltantes":
            columna = variable_faltantes
            cantidad = int(limpio[columna].isna().sum())
            if estrategia == "Eliminar filas con faltantes":
                limpio = limpio.dropna(subset=[columna]).copy()
                acciones.append(
                    f"Se eliminaron {cantidad:,} filas con faltantes en {columna}."
                )
            elif estrategia == "Completar con la mediana":
                relleno = limpio[columna].median()
                limpio[columna] = limpio[columna].fillna(relleno)
                acciones.append(
                    f"Se completaron {cantidad:,} faltantes de {columna} con la mediana ({relleno:g})."
                )
            elif estrategia == "Completar con el valor más frecuente":
                moda = limpio[columna].mode(dropna=True)
                if not moda.empty:
                    relleno = moda.iloc[0]
                    limpio[columna] = limpio[columna].fillna(relleno)
                    acciones.append(
                        f"Se completaron {cantidad:,} faltantes de {columna} con el valor más frecuente ({relleno})."
                    )

        if not acciones:
            acciones.append(
                "No se hicieron cambios; se conservaron los faltantes y duplicados."
            )
        st.session_state.df_limpio = limpio
        st.session_state.limpieza_acciones = acciones
        st.success("Se actualizó la copia de trabajo.")

    df_limpio = st.session_state.df_limpio
    st.subheader("Resultado y bitácora")
    r1, r2, r3 = st.columns(3)
    r1.metric("Filas de trabajo", f"{len(df_limpio):,}")
    r2.metric("Faltantes restantes", f"{int(df_limpio.isna().sum().sum()):,}")
    r3.metric("Duplicados restantes", f"{int(df_limpio.duplicated().sum()):,}")
    for accion in st.session_state.limpieza_acciones:
        st.write(f"- {accion}")

    csv = df_limpio.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar copia de trabajo (CSV)",
        data=csv,
        file_name="infraestructura_ti_limpia.csv",
        mime="text/csv",
        key="descargar_limpieza",
    )
