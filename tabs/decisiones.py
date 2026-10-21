import pandas as pd
import streamlit as st


def _numero(df: pd.DataFrame, columna: str) -> pd.Series:
    if columna not in df:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.to_numeric(df[columna], errors="coerce")


def _formato_numero(valor: float, columna: str) -> str:
    if pd.isna(valor):
        return "N/D"
    if "costo" in columna.casefold() or "usd" in columna.casefold():
        return f"${valor:,.0f}"
    if "disponibilidad" in columna.casefold() or "%" in columna:
        return f"{valor:,.2f}%"
    return f"{valor:,.2f}"


def mostrar(df: pd.DataFrame) -> None:
    st.header("6. Plan de revisión")
    st.markdown(
        "Convierte las prioridades sugeridas en una lista de trabajo revisable. "
        "La descarga no ejecuta cambios ni sustituye la aprobación del dueño "
        "del servicio."
    )

    recomendaciones = st.session_state.get("recomendaciones")
    if recomendaciones is None or recomendaciones.empty:
        st.info("No hay prioridades todavía. Revisa primero la pestaña Prescriptiva.")
        return

    columna_prioridad = "segmento_recomendado"
    if columna_prioridad not in recomendaciones:
        st.error("No se encontró la prioridad generada en la pestaña Prescriptiva.")
        return

    filas = []
    for prioridad, grupo in recomendaciones.groupby(columna_prioridad, dropna=False):
        fila = {
            "Prioridad sugerida": prioridad,
            "Clientes / registros": len(grupo),
            "Participación": len(grupo) / len(recomendaciones),
            "Disponibilidad promedio (%)": _numero(grupo, "Disponibilidad_%").mean(),
            "Incidentes mensuales promedio": _numero(grupo, "Incidentes_Mensuales").mean(),
            "Tickets abiertos promedio": _numero(grupo, "Tickets_Abiertos").mean(),
            "Acción sugerida": grupo["accion_recomendada"].iat[0],
            "Responsable sugerido": grupo["responsable_sugerido"].iat[0],
        }
        filas.append(fila)
    resumen = pd.DataFrame(filas).sort_values("Prioridad sugerida")

    p1, p2, p3 = st.columns(3)
    p1.metric("Clientes / registros", f"{len(recomendaciones):,}")
    p2.metric(
        "Prioridad P1",
        f"{int(recomendaciones[columna_prioridad].str.startswith('P1').sum()):,}",
    )
    p3.metric(
        "Disponibilidad promedio",
        f"{_numero(recomendaciones, 'Disponibilidad_%').mean():.2f}%"
        if _numero(recomendaciones, "Disponibilidad_%").notna().any()
        else "N/D",
    )

    st.subheader("Resumen por prioridad")
    st.dataframe(
        resumen.style.format(
            {
                "Participación": "{:.1%}",
                "Disponibilidad promedio (%)": "{:.2f}%",
                "Incidentes mensuales promedio": "{:.1f}",
                "Tickets abiertos promedio": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Los promedios describen los registros agrupados y no estiman el efecto "
        "de las acciones propuestas."
    )

    st.subheader("Preparar una revisión")
    prioridades = resumen["Prioridad sugerida"].tolist()
    prioridad_elegida = st.selectbox(
        "Prioridad para revisar primero",
        prioridades,
        key="prioridad_decisiones_infraestructura",
    )
    grupo = recomendaciones.loc[
        recomendaciones[columna_prioridad].eq(prioridad_elegida)
    ].copy()
    primera_fila = grupo.iloc[0]
    a, b = st.columns(2)
    a.metric("Tamaño de la lista", f"{len(grupo):,}")
    b.write(f"**Responsable sugerido:** {primera_fila['responsable_sugerido']}")
    st.write(f"**Acción propuesta:** {primera_fila['accion_recomendada']}")
    st.write(f"**Evidencia de la regla:** {primera_fila['evidencia_regla']}")

    campos_muestra = [
        columna
        for columna in [
            "Cliente", "Fecha", "Sector", "Criticidad", "Disponibilidad_%",
            "Incidentes_Mensuales", "Tickets_Abiertos", "Estado_Proyecto",
            "evidencia_regla", "accion_recomendada", "responsable_sugerido",
        ]
        if columna in grupo
    ]
    st.dataframe(grupo[campos_muestra], use_container_width=True, hide_index=True)

    confirmar_revision = st.checkbox(
        "Confirmo que el responsable validará SLA, evidencia, riesgos y ventana de cambio antes de ejecutar una intervención.",
        key="confirmar_revision_decisiones",
    )
    columnas_exportacion = [
        columna
        for columna in [
            "Cliente", "Fecha", "Sector", "Criticidad", "Modelo_Operacion",
            "Servidores_Fisicos", "Maquinas_Virtuales", "Almacenamiento_TB",
            "Disponibilidad_%", "Incidentes_Mensuales", "Tickets_Abiertos",
            "Costo_TI_Anual_USD", "Estado_Proyecto", "segmento_recomendado",
            "evidencia_regla", "accion_recomendada", "responsable_sugerido",
            "siguiente_paso", "umbral_disponibilidad", "percentil_carga",
        ]
        if columna in recomendaciones
    ]
    st.download_button(
        "Descargar lista de revisión (CSV)",
        data=recomendaciones[columnas_exportacion]
        .sort_values("segmento_recomendado")
        .to_csv(index=False)
        .encode("utf-8-sig"),
        file_name="plan_revision_infraestructura_ti.csv",
        mime="text/csv",
        disabled=not confirmar_revision,
        help="Requiere confirmar la revisión técnica previa a cualquier cambio.",
        key="descargar_plan_revision_infraestructura",
    )

    st.divider()
    st.subheader("Conclusión e insights del proyecto")

    clientes_analizados = (
        int(df["Cliente"].nunique()) if "Cliente" in df else len(df)
    )
    disponibilidad = _numero(df, "Disponibilidad_%").dropna()
    incidentes = _numero(df, "Incidentes_Mensuales").dropna()
    faltantes = int(df.isna().sum().sum())
    duplicados = int(df.duplicated().sum())
    disponibilidad_texto = (
        f"{disponibilidad.mean():.2f}%" if not disponibilidad.empty else "sin datos"
    )
    incidentes_texto = (
        f"{incidentes.mean():.1f} por registro" if not incidentes.empty else "sin datos"
    )

    prioridad_alta = int(
        recomendaciones[columna_prioridad].str.startswith(("P1", "P2")).sum()
    )
    participacion_alta = prioridad_alta / len(recomendaciones) if len(recomendaciones) else 0
    prioridades = recomendaciones[columna_prioridad].value_counts()
    prioridad_mas_comun = prioridades.index[0] if not prioridades.empty else "sin prioridad"

    st.markdown(
        f"El análisis actual reúne **{len(df):,} registros de {clientes_analizados:,} clientes**. "
        f"La disponibilidad promedio observada es **{disponibilidad_texto}** y el promedio "
        f"de incidentes es **{incidentes_texto}**. "
        f"La regla prescriptiva asignó prioridad alta (P1 o P2) a **{prioridad_alta:,} de "
        f"{len(recomendaciones):,} clientes ({participacion_alta:.1%})**; la categoría más "
        f"numerosa es **{prioridad_mas_comun}**."
    )

    sector_top = None
    if "Sector" in recomendaciones and recomendaciones["Sector"].notna().any():
        sector_top = recomendaciones["Sector"].value_counts().index[0]
        sector_count = int(recomendaciones["Sector"].value_counts().iloc[0])
        st.write(
            f"En la vista más reciente por cliente, el sector más representado es "
            f"**{sector_top}** ({sector_count} clientes)."
        )

    resumen_predictivo = st.session_state.get("resumen_predictivo")
    if resumen_predictivo:
        error_modelo = resumen_predictivo["mae"]
        error_referencia = resumen_predictivo["mae_base"]
        if error_referencia > 0:
            cambio_error = (error_referencia - error_modelo) / error_referencia
        else:
            cambio_error = float("nan")
        if pd.notna(cambio_error) and cambio_error > 0:
            lectura_modelo = (
                f"El modelo redujo el error absoluto medio en **{cambio_error:.1%}** "
                "frente a la referencia del promedio"
            )
        else:
            lectura_modelo = (
                "El modelo no mejoró el error frente a la referencia del promedio"
            )
        objetivo = resumen_predictivo["objetivo"]
        r2 = resumen_predictivo["r2"]
        r2_texto = f"{r2:.3f}" if pd.notna(r2) else "no disponible"
        st.write(
            f"Para **{objetivo}**, {lectura_modelo} en una evaluación reservada por cliente "
            f"(MAE {_formato_numero(error_modelo, objetivo)}, R² {r2_texto}); se evaluaron "
            f"{resumen_predictivo['prueba_clientes']} clientes y "
            f"{resumen_predictivo['prueba_registros']} registros."
        )
    else:
        st.write(
            "No hay una evaluación predictiva disponible en esta sesión; revisa "
            "la pestaña Predictiva para interpretar el desempeño del modelo."
        )

    st.markdown(
        f"En la copia de trabajo quedan **{faltantes:,} celdas faltantes** y "
        f"**{duplicados:,} filas duplicadas exactas**. Antes de convertir estas "
        "prioridades en cambios, valida los valores con los SLA, confirma el contexto "
        "operativo y acuerda responsables y ventanas de mantenimiento. La evaluación "
        "predictiva es una prueba no temporal en una muestra pequeña; las reglas "
        "prescriptivas organizan la revisión, pero no cuantifican el beneficio de intervenir."
    )
