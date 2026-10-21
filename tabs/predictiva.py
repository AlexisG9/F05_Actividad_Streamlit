import numpy as np
import pandas as pd
import streamlit as st


OBJETIVOS_CANDIDATOS = [
    "Disponibilidad_%",
    "Incidentes_Mensuales",
    "Tickets_Abiertos",
    "Costo_TI_Anual_USD",
]


@st.cache_data(show_spinner="Entrenando y evaluando el modelo...")
def _entrenar_evaluar(df: pd.DataFrame, objetivo: str) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    datos = df.copy()
    y = pd.to_numeric(datos[objetivo], errors="coerce")
    mascara = y.notna() & np.isfinite(y)
    datos = datos.loc[mascara].copy()
    y = y.loc[mascara].astype(float)

    excluir = {objetivo, "Cliente", "Fecha", "cliente_id", "fecha_alta"}
    columnas = [columna for columna in datos.columns if columna not in excluir]
    if not columnas:
        raise ValueError("No hay variables disponibles para explicar el indicador elegido.")

    X = datos[columnas].copy()
    numericas = X.select_dtypes(include="number").columns.tolist()
    categoricas = [columna for columna in X.columns if columna not in numericas]
    for columna in numericas:
        X[columna] = pd.to_numeric(X[columna], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    transformadores = []
    if numericas:
        transformadores.append(
            (
                "numericas",
                Pipeline([("completar", SimpleImputer(strategy="median", add_indicator=True))]),
                numericas,
            )
        )
    if categoricas:
        transformadores.append(
            (
                "categoricas",
                Pipeline(
                    [
                        ("completar", SimpleImputer(strategy="constant", fill_value="(sin dato)")),
                        ("codificar", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categoricas,
            )
        )
    preparar = ColumnTransformer(transformadores, remainder="drop")
    modelo = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preparar", preparar), ("modelo", modelo)])

    if "Cliente" in datos:
        grupos = datos["Cliente"].astype("string")
        faltan_grupo = grupos.isna()
        grupos = grupos.fillna(pd.Series(datos.index.astype(str), index=datos.index))
        if grupos.nunique() < 2:
            raise ValueError("Se necesitan al menos dos clientes distintos para separar la evaluación.")
        separador = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        indices_entrenamiento, indices_prueba = next(
            separador.split(X, y, groups=grupos)
        )
        grupos_entrenamiento = grupos.iloc[indices_entrenamiento]
        grupos_prueba = grupos.iloc[indices_prueba]
        clientes_entrenamiento = int(grupos_entrenamiento.nunique())
        clientes_prueba = int(grupos_prueba.nunique())
        separacion = "por cliente"
        grupos_faltantes = int(faltan_grupo.sum())
    else:
        if len(datos) < 8:
            raise ValueError("Se requieren al menos ocho registros válidos para evaluar el modelo.")
        indices_entrenamiento, indices_prueba = train_test_split(
            np.arange(len(datos)), test_size=0.25, random_state=42
        )
        clientes_entrenamiento = len(indices_entrenamiento)
        clientes_prueba = len(indices_prueba)
        separacion = "por registro (no existe una columna Cliente)"
        grupos_faltantes = 0

    if len(indices_entrenamiento) < 4 or len(indices_prueba) < 2:
        raise ValueError("La separación dejó muy pocos registros para una evaluación útil.")

    X_entrenamiento = X.iloc[indices_entrenamiento]
    X_prueba = X.iloc[indices_prueba]
    y_entrenamiento = y.iloc[indices_entrenamiento]
    y_prueba = y.iloc[indices_prueba]
    if y_entrenamiento.nunique() < 2:
        raise ValueError("El indicador no tiene variación suficiente en los datos de entrenamiento.")

    pipeline.fit(X_entrenamiento, y_entrenamiento)
    predicciones = pipeline.predict(X_prueba)
    prediccion_base = np.full(len(y_prueba), y_entrenamiento.mean())

    r2 = float(r2_score(y_prueba, predicciones)) if y_prueba.nunique() > 1 else float("nan")
    comparacion = pd.DataFrame(
        {
            "Valor observado": y_prueba.to_numpy(),
            "Estimación del modelo": predicciones,
            "Estimación base (promedio)": prediccion_base,
        }
    )
    comparacion["Error absoluto"] = (
        comparacion["Valor observado"] - comparacion["Estimación del modelo"]
    ).abs()
    comparacion = comparacion.sort_values("Error absoluto", ascending=False)

    return {
        "objetivo": objetivo,
        "mae": float(mean_absolute_error(y_prueba, predicciones)),
        "rmse": float(np.sqrt(mean_squared_error(y_prueba, predicciones))),
        "r2": r2,
        "mae_base": float(mean_absolute_error(y_prueba, prediccion_base)),
        "entrenamiento_registros": len(indices_entrenamiento),
        "prueba_registros": len(indices_prueba),
        "entrenamiento_clientes": clientes_entrenamiento,
        "prueba_clientes": clientes_prueba,
        "separacion": separacion,
        "grupos_faltantes": grupos_faltantes,
        "comparacion": comparacion,
        "variables": columnas,
    }


def _formato_valor(valor: float, objetivo: str) -> str:
    if pd.isna(valor):
        return "N/D"
    if "costo" in objetivo.casefold() or "usd" in objetivo.casefold():
        return f"${valor:,.0f}"
    if "disponibilidad" in objetivo.casefold() or "%" in objetivo:
        return f"{valor:,.2f}%"
    return f"{valor:,.2f}"


def mostrar(df: pd.DataFrame) -> None:
    st.session_state.pop("resumen_predictivo", None)
    st.header("4. Predicción de indicadores de infraestructura")
    st.markdown(
        "Entrena un modelo para estimar un indicador operativo a partir de las "
        "características disponibles de cada registro. La evaluación reserva "
        "clientes completos para comprobar cómo generaliza a clientes que el "
        "modelo no vio durante el entrenamiento."
    )

    if df.empty:
        st.warning("No hay registros para entrenar un modelo.")
        return

    objetivos = [
        columna
        for columna in OBJETIVOS_CANDIDATOS
        if columna in df and pd.to_numeric(df[columna], errors="coerce").notna().any()
    ]
    if not objetivos:
        st.error(
            "No se encontró un indicador numérico disponible. Se esperaba al menos "
            "uno de: disponibilidad, incidentes, tickets abiertos o costo anual de TI."
        )
        return

    st.info(
        "La salida es una estimación del indicador seleccionado, no una decisión "
        "automática. El modelo no predice una fecha futura: evalúa registros de "
        "clientes reservados y puede servir como ejercicio de generalización."
    )
    objetivo = st.selectbox(
        "¿Qué indicador quieres estimar?",
        options=objetivos,
        index=objetivos.index("Disponibilidad_%") if "Disponibilidad_%" in objetivos else 0,
        format_func=lambda nombre: {
            "Disponibilidad_%": "Disponibilidad (%)",
            "Incidentes_Mensuales": "Incidentes mensuales",
            "Tickets_Abiertos": "Tickets abiertos",
            "Costo_TI_Anual_USD": "Costo anual de TI (USD)",
        }.get(nombre, nombre),
        key="objetivo_predictivo_infraestructura",
    )

    try:
        resultado = _entrenar_evaluar(df, objetivo)
    except ModuleNotFoundError:
        st.error("Falta scikit-learn. Instala las dependencias del proyecto para usar esta sección.")
        return
    except ValueError as error:
        st.error(f"No se pudo evaluar el modelo: {error}")
        return

    st.session_state.resumen_predictivo = {
        "objetivo": objetivo,
        "mae": resultado["mae"],
        "mae_base": resultado["mae_base"],
        "r2": resultado["r2"],
        "prueba_registros": resultado["prueba_registros"],
        "prueba_clientes": resultado["prueba_clientes"],
    }

    st.subheader("Desempeño en clientes reservados")
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Error absoluto medio (MAE)",
        _formato_valor(resultado["mae"], objetivo),
        help="Promedio de la diferencia absoluta entre el valor estimado y el observado. Menor es mejor.",
    )
    m2.metric(
        "Referencia simple (MAE)",
        _formato_valor(resultado["mae_base"], objetivo),
        help="Error al estimar siempre el promedio del indicador en los datos de entrenamiento.",
    )
    m3.metric(
        "R² en evaluación",
        f"{resultado['r2']:.3f}" if pd.notna(resultado["r2"]) else "N/D",
        help="Compara el modelo con una referencia basada en el promedio de los valores observados. Puede ser negativo.",
    )

    diferencia_mae = resultado["mae_base"] - resultado["mae"]
    if diferencia_mae > 0:
        st.success(
            f"En esta separación, el modelo redujo el MAE en "
            f"{_formato_valor(diferencia_mae, objetivo)} frente a estimar siempre el promedio."
        )
    else:
        st.warning(
            "En esta separación, el modelo no superó la referencia simple del promedio. "
            "Conviene revisar las variables, la calidad de los datos y el tamaño de la muestra."
        )

    st.subheader("Estimaciones frente a valores observados")
    comparacion = resultado["comparacion"]
    grafica = comparacion[["Valor observado", "Estimación del modelo"]]
    st.scatter_chart(grafica, x="Valor observado", y="Estimación del modelo")
    tabla = comparacion.head(20).copy()
    for columna in ["Valor observado", "Estimación del modelo", "Estimación base (promedio)", "Error absoluto"]:
        tabla[columna] = tabla[columna].map(lambda valor: _formato_valor(valor, objetivo))
    st.dataframe(tabla, use_container_width=True, hide_index=True)
    st.caption("La tabla muestra hasta 20 casos con mayor error absoluto para facilitar su revisión.")

    with st.expander("Cómo se construyó la evaluación"):
        st.markdown(
            f"- Indicador estimado: **{objetivo}**.\n"
            f"- Se usaron **{resultado['entrenamiento_registros']:,} registros** para entrenar y "
            f"**{resultado['prueba_registros']:,}** para evaluar. La separación fue {resultado['separacion']}.\n"
            f"- Grupos de entrenamiento y evaluación: **{resultado['entrenamiento_clientes']:,}** y "
            f"**{resultado['prueba_clientes']:,}**, respectivamente.\n"
            "- Se excluyeron `Cliente` y `Fecha` como predictores para evitar memorizar "
            "identidades o usar el calendario como sustituto de una explicación operativa. "
            "Los faltantes de las demás variables se imputan dentro del proceso de entrenamiento.\n"
            "- La referencia simple predice el promedio del indicador de entrenamiento. "
            "El MAE y R² comparan el modelo con valores reservados.\n"
            "- Esta evaluación no es una validación temporal ni una garantía de desempeño "
            "en producción. Con una muestra pequeña, los resultados pueden cambiar bastante "
            "según los clientes reservados."
        )
        if resultado["grupos_faltantes"]:
            st.caption(
                f"{resultado['grupos_faltantes']} registros sin nombre de cliente se trataron "
                "como grupos individuales durante la separación."
            )
        st.caption(
            "Variables consideradas: " + ", ".join(resultado["variables"])
        )
