import pandas as pd
import streamlit as st


COLUMNAS_OPERATIVAS = [
    "Disponibilidad_%",
    "Incidentes_Mensuales",
    "Tickets_Abiertos",
]


def _numero(df: pd.DataFrame, columna: str) -> pd.Series:
    if columna not in df:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.to_numeric(df[columna], errors="coerce")


def _ultima_observacion_por_cliente(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva una fila por cliente, usando la fecha más reciente disponible."""
    resultado = df.copy().reset_index(drop=True)
    if "Cliente" not in resultado:
        return resultado

    resultado["_orden_origen"] = range(len(resultado))
    if "Fecha" in resultado:
        resultado["_fecha_orden"] = pd.to_datetime(resultado["Fecha"], errors="coerce")
        resultado = resultado.sort_values(
            ["_fecha_orden", "_orden_origen"], kind="stable", na_position="first"
        )

    # Las filas sin identificador se mantienen como observaciones independientes.
    resultado["_cliente_orden"] = resultado["Cliente"].astype("string")
    sin_cliente = resultado["_cliente_orden"].isna()
    resultado.loc[sin_cliente, "_cliente_orden"] = (
        "__registro_sin_cliente__" + resultado.loc[sin_cliente, "_orden_origen"].astype(str)
    )
    resultado = resultado.drop_duplicates("_cliente_orden", keep="last")
    return resultado.drop(
        columns=["_orden_origen", "_fecha_orden", "_cliente_orden"], errors="ignore"
    ).reset_index(drop=True)


def construir_recomendaciones(
    df: pd.DataFrame,
    disponibilidad_minima: float = 97.0,
    percentil_carga: int = 75,
) -> pd.DataFrame:
    """Genera prioridades operativas explicables; no modifica el dataset fuente."""
    resultado = _ultima_observacion_por_cliente(df)
    disponibilidad = _numero(resultado, "Disponibilidad_%")
    incidentes = _numero(resultado, "Incidentes_Mensuales")
    tickets = _numero(resultado, "Tickets_Abiertos")

    corte_incidentes = (
        float(incidentes.dropna().quantile(percentil_carga / 100))
        if incidentes.notna().any()
        else float("inf")
    )
    corte_tickets = (
        float(tickets.dropna().quantile(percentil_carga / 100))
        if tickets.notna().any()
        else float("inf")
    )
    baja_disponibilidad = disponibilidad.notna() & disponibilidad.lt(disponibilidad_minima)
    alta_carga_incidentes = incidentes.notna() & incidentes.gt(corte_incidentes)
    alta_carga_tickets = tickets.notna() & tickets.gt(corte_tickets)
    degradacion = baja_disponibilidad | alta_carga_incidentes | alta_carga_tickets

    if "Criticidad" in resultado:
        criticidad_alta = (
            resultado["Criticidad"].astype("string").str.casefold().eq("alta").fillna(False)
        )
    else:
        criticidad_alta = pd.Series(False, index=resultado.index)

    columnas_operativas = [c for c in COLUMNAS_OPERATIVAS if c in resultado]
    faltan_metricas = (
        resultado[columnas_operativas].isna().any(axis=1)
        if columnas_operativas
        else pd.Series(True, index=resultado.index)
    )
    if "Estado_Proyecto" in resultado:
        estado = resultado["Estado_Proyecto"].astype("string").str.casefold()
        proyecto_en_curso = estado.notna() & ~estado.isin(["operación", "operacion"])
    else:
        proyecto_en_curso = pd.Series(False, index=resultado.index)

    prioridad_1 = criticidad_alta & degradacion
    prioridad_2 = ~prioridad_1 & degradacion
    prioridad_3 = ~prioridad_1 & ~prioridad_2 & (criticidad_alta | faltan_metricas)
    prioridad_4 = ~prioridad_1 & ~prioridad_2 & ~prioridad_3 & proyecto_en_curso

    prioridades = pd.Series("P5 · Monitoreo rutinario", index=resultado.index, dtype="object")
    prioridades.loc[prioridad_4] = "P4 · Seguimiento de proyecto"
    prioridades.loc[prioridad_3] = "P3 · Validar servicio y datos"
    prioridades.loc[prioridad_2] = "P2 · Reducir carga operativa"
    prioridades.loc[prioridad_1] = "P1 · Estabilizar servicio crítico"

    def _evidencia(indice: int) -> str:
        señales = []
        if baja_disponibilidad.loc[indice]:
            señales.append(f"disponibilidad {disponibilidad.loc[indice]:.2f}% bajo el umbral de {disponibilidad_minima:.1f}%")
        if alta_carga_incidentes.loc[indice]:
            señales.append(f"incidentes por encima del percentil {percentil_carga} ({corte_incidentes:.1f})")
        if alta_carga_tickets.loc[indice]:
            señales.append(f"tickets abiertos por encima del percentil {percentil_carga} ({corte_tickets:.1f})")
        if criticidad_alta.loc[indice]:
            señales.append("criticidad alta")
        if faltan_metricas.loc[indice]:
            señales.append("faltan datos de una o más métricas operativas")
        if proyecto_en_curso.loc[indice]:
            señales.append("hay un proyecto distinto de operación en curso")
        return "; ".join(señales) if señales else "Sin señales de alerta en las reglas seleccionadas."

    acciones = {
        "P1 · Estabilizar servicio crítico": "Revisar disponibilidad y carga operativa; acordar respuesta con continuidad y operaciones.",
        "P2 · Reducir carga operativa": "Analizar incidentes y tickets recurrentes; priorizar causas y capacidad de atención.",
        "P3 · Validar servicio y datos": "Confirmar el SLA, la criticidad y las métricas faltantes antes de planear cambios.",
        "P4 · Seguimiento de proyecto": "Revisar dependencias, riesgos y ventana de cambio del proyecto registrado.",
        "P5 · Monitoreo rutinario": "Mantener el monitoreo y revisar tendencias en el siguiente ciclo operativo.",
    }
    responsables = {
        "P1 · Estabilizar servicio crítico": "Operaciones de TI / Continuidad",
        "P2 · Reducir carga operativa": "Operaciones de TI / Gestión de servicios",
        "P3 · Validar servicio y datos": "Dueño del servicio / Gobierno de datos",
        "P4 · Seguimiento de proyecto": "Líder del proyecto / Arquitectura TI",
        "P5 · Monitoreo rutinario": "Operaciones de TI",
    }
    resultado["segmento_recomendado"] = prioridades
    resultado["evidencia_regla"] = [_evidencia(indice) for indice in resultado.index]
    resultado["accion_recomendada"] = resultado["segmento_recomendado"].map(acciones)
    resultado["responsable_sugerido"] = resultado["segmento_recomendado"].map(responsables)
    resultado["siguiente_paso"] = "Validar evidencia y prioridad con el responsable del servicio antes de ejecutar cambios."
    resultado["riesgo_medicion"] = "Regla heurística con umbrales configurables; no estima impacto ni causalidad."
    resultado["umbral_disponibilidad"] = disponibilidad_minima
    resultado["percentil_carga"] = percentil_carga
    resultado["corte_incidentes"] = corte_incidentes
    resultado["corte_tickets"] = corte_tickets
    return resultado


def mostrar(df: pd.DataFrame) -> None:
    st.header("5. Recomendaciones operativas")
    st.markdown(
        "Convierte señales del dataset en una lista explicable de revisión. "
        "La prioridad ayuda a organizar el análisis; no reemplaza un SLA, una "
        "evaluación técnica ni la aprobación de cambios."
    )
    if df.empty:
        st.warning("No hay registros para priorizar.")
        return

    disponibilidad_disponible = "Disponibilidad_%" in df
    controles = st.columns(2)
    disponibilidad_minima = controles[0].slider(
        "Umbral de disponibilidad para el ejercicio (%)",
        min_value=90.0,
        max_value=100.0,
        value=97.0,
        step=0.5,
        disabled=not disponibilidad_disponible,
        help="Es un umbral de análisis configurable, no un SLA inferido del dataset.",
        key="umbral_disponibilidad_prescriptiva",
    )
    percentil_carga = controles[1].slider(
        "Percentil para señalar carga alta de incidentes y tickets",
        min_value=50,
        max_value=95,
        value=75,
        step=5,
        help="Señala valores estrictamente por encima del percentil elegido en los registros analizados.",
        key="percentil_carga_prescriptiva",
    )

    recomendaciones = construir_recomendaciones(
        df,
        disponibilidad_minima=disponibilidad_minima,
        percentil_carga=percentil_carga,
    )
    st.session_state.recomendaciones = recomendaciones
    st.session_state.parametros_prescriptiva = {
        "disponibilidad_minima": disponibilidad_minima,
        "percentil_carga": percentil_carga,
    }

    cantidad_clientes = (
        int(recomendaciones["Cliente"].nunique())
        if "Cliente" in recomendaciones
        else len(recomendaciones)
    )
    alertas = recomendaciones["segmento_recomendado"].str.startswith(("P1", "P2"))
    k1, k2, k3 = st.columns(3)
    k1.metric("Clientes / registros evaluados", f"{cantidad_clientes:,}")
    k2.metric("Prioridad alta (P1 y P2)", f"{int(alertas.sum()):,}")
    k3.metric("Indicadores con datos faltantes", f"{int(recomendaciones['evidencia_regla'].str.contains('faltan datos').sum()):,}")
    if "Cliente" in df and len(df) != len(recomendaciones):
        st.caption(
            "Se conservó la observación más reciente por cliente para evitar "
            "duplicar recomendaciones entre fechas."
        )

    conteos = (
        recomendaciones["segmento_recomendado"]
        .value_counts()
        .rename_axis("Prioridad sugerida")
        .rename("Clientes")
        .reset_index()
    )
    st.subheader("Distribución de prioridades")
    st.bar_chart(conteos, x="Prioridad sugerida", y="Clientes")
    st.dataframe(conteos, use_container_width=True, hide_index=True)

    st.subheader("Reglas de priorización")
    st.dataframe(
        pd.DataFrame(
            [
                {"Orden": 1, "Condición": "Criticidad alta y al menos una señal operativa", "Prioridad": "P1 · Estabilizar servicio crítico"},
                {"Orden": 2, "Condición": "Disponibilidad bajo umbral o incidentes / tickets sobre el percentil elegido", "Prioridad": "P2 · Reducir carga operativa"},
                {"Orden": 3, "Condición": "Criticidad alta sin alerta, o falta algún indicador operativo", "Prioridad": "P3 · Validar servicio y datos"},
                {"Orden": 4, "Condición": "Proyecto registrado distinto de Operación", "Prioridad": "P4 · Seguimiento de proyecto"},
                {"Orden": 5, "Condición": "Ninguna de las condiciones anteriores", "Prioridad": "P5 · Monitoreo rutinario"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Umbral configurado: {disponibilidad_minima:.1f}% de disponibilidad; "
        f"los cortes del percentil {percentil_carga} son "
        f"{recomendaciones['corte_incidentes'].iloc[0]:.1f} incidentes y "
        f"{recomendaciones['corte_tickets'].iloc[0]:.1f} tickets. Confirma los "
        "umbrales con los SLA y responsables del servicio."
    )

    st.subheader("Revisar recomendaciones")
    prioridad = st.selectbox(
        "Filtrar por prioridad",
        options=conteos["Prioridad sugerida"].tolist(),
        key="prioridad_muestra_prescriptiva",
    )
    columnas = [
        columna
        for columna in [
            "Cliente", "Fecha", "Sector", "Criticidad", "Disponibilidad_%",
            "Incidentes_Mensuales", "Tickets_Abiertos", "Estado_Proyecto",
            "segmento_recomendado", "evidencia_regla", "accion_recomendada",
            "responsable_sugerido",
        ]
        if columna in recomendaciones
    ]
    muestra = recomendaciones.loc[
        recomendaciones["segmento_recomendado"].eq(prioridad), columnas
    ].head(25)
    st.dataframe(muestra, use_container_width=True, hide_index=True)
    
