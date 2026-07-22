"""
Dashboard interactivo de estadísticas de accidentes de tránsito en Bogotá.

Ejecutar con:
    streamlit run app.py

El dashboard funciona de dos formas:
1. Si el usuario carga un archivo CSV propio (por ejemplo, extraído del portal
   de datos abiertos de Bogotá / SIMUR - Secretaría Distrital de Movilidad),
   la app usa esos datos reales.
2. Si no se carga ningún archivo, la app genera un set de datos simulado
   pero realista (mismas localidades, tipos de accidente, gravedad, etc.)
   para que el usuario pueda explorar el dashboard de inmediato.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Accidentalidad Vial - Bogotá",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOCALIDADES = [
    "Usaquén", "Chapinero", "Santa Fe", "San Cristóbal", "Usme",
    "Tunjuelito", "Bosa", "Kennedy", "Fontibón", "Engativá",
    "Suba", "Barrios Unidos", "Teusaquillo", "Los Mártires",
    "Antonio Nariño", "Puente Aranda", "La Candelaria",
    "Rafael Uribe Uribe", "Ciudad Bolívar", "Sumapaz",
]

# Coordenadas aproximadas del centroide de cada localidad (para el mapa)
COORDS_LOCALIDAD = {
    "Usaquén": (4.6946, -74.0300), "Chapinero": (4.6486, -74.0625),
    "Santa Fe": (4.6080, -74.0757), "San Cristóbal": (4.5570, -74.0817),
    "Usme": (4.4790, -74.1260), "Tunjuelito": (4.5720, -74.1330),
    "Bosa": (4.6180, -74.1770), "Kennedy": (4.6280, -74.1560),
    "Fontibón": (4.6740, -74.1460), "Engativá": (4.7100, -74.1120),
    "Suba": (4.7480, -74.0930), "Barrios Unidos": (4.6670, -74.0840),
    "Teusaquillo": (4.6360, -74.0930), "Los Mártires": (4.6040, -74.0910),
    "Antonio Nariño": (4.5860, -74.1000), "Puente Aranda": (4.6180, -74.1150),
    "La Candelaria": (4.5960, -74.0750), "Rafael Uribe Uribe": (4.5580, -74.1050),
    "Ciudad Bolívar": (4.5000, -74.1580), "Sumapaz": (4.2200, -74.3800),
}

TIPOS_ACCIDENTE = ["Choque", "Atropello", "Volcamiento", "Caída de ocupante", "Incendio", "Otro"]
GRAVEDAD = ["Solo daños", "Con heridos", "Con muertos"]
CLASES_VEHICULO = ["Automóvil", "Motocicleta", "Bus/Buseta", "Bicicleta", "Camión", "Peatón"]
CAUSAS = [
    "Exceso de velocidad", "No respetar semáforo/señal", "Distracción del conductor",
    "Embriaguez / sustancias", "Adelantamiento indebido", "Condiciones de la vía",
    "Fallas mecánicas", "No mantener distancia",
]

MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


@st.cache_data
def generar_datos_simulados(n=9000, seed=42):
    """Genera un dataset sintético pero estadísticamente plausible de accidentes."""
    rng = np.random.default_rng(seed)

    fechas = pd.date_range("2021-01-01", "2025-12-31", freq="h")
    fechas = rng.choice(fechas, size=n, replace=True)
    fechas = pd.to_datetime(fechas)

    # Pesos de localidad (Kennedy, Suba, Engativá suelen ser las de mayor volumen vial)
    pesos_loc = np.array([
        6, 5, 3, 4, 3, 3, 7, 11, 6, 9, 10, 4, 4, 3, 2, 6, 1, 5, 7, 0.3
    ])
    pesos_loc = pesos_loc / pesos_loc.sum()
    localidad = rng.choice(LOCALIDADES, size=n, p=pesos_loc)

    tipo = rng.choice(TIPOS_ACCIDENTE, size=n, p=[0.45, 0.18, 0.10, 0.12, 0.03, 0.12])
    gravedad = rng.choice(GRAVEDAD, size=n, p=[0.72, 0.25, 0.03])
    clase_vehiculo = rng.choice(CLASES_VEHICULO, size=n, p=[0.30, 0.32, 0.10, 0.09, 0.07, 0.12])
    causa = rng.choice(CAUSAS, size=n)

    heridos = np.where(gravedad == "Con heridos", rng.integers(1, 4, size=n), 0)
    muertos = np.where(gravedad == "Con muertos", rng.integers(1, 2, size=n), 0)

    lat_base = np.array([COORDS_LOCALIDAD[loc][0] for loc in localidad])
    lon_base = np.array([COORDS_LOCALIDAD[loc][1] for loc in localidad])
    lat = lat_base + rng.normal(0, 0.012, size=n)
    lon = lon_base + rng.normal(0, 0.012, size=n)

    df = pd.DataFrame({
        "fecha": fechas,
        "localidad": localidad,
        "tipo_accidente": tipo,
        "gravedad": gravedad,
        "clase_vehiculo": clase_vehiculo,
        "causa_probable": causa,
        "heridos": heridos,
        "muertos": muertos,
        "latitud": lat,
        "longitud": lon,
    })
    df["año"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["mes_nombre"] = df["mes"].apply(lambda m: MESES_ES[m - 1])
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["dia_semana_nombre"] = df["dia_semana"].apply(lambda d: DIAS_ES[d])
    df["hora"] = df["fecha"].dt.hour
    return df.sort_values("fecha").reset_index(drop=True)


@st.cache_data
def cargar_csv(archivo):
    df = pd.read_csv(archivo)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["año"] = df["fecha"].dt.year
        df["mes"] = df["fecha"].dt.month
        df["dia_semana"] = df["fecha"].dt.dayofweek
        df["hora"] = df["fecha"].dt.hour
    return df


# --------------------------------------------------------------------------
# BARRA LATERAL
# --------------------------------------------------------------------------
st.sidebar.title("🚦 Panel de control")
archivo = st.sidebar.file_uploader(
    "Carga tu propio CSV de accidentes (opcional)", type=["csv"]
)

if archivo is not None:
    df = cargar_csv(archivo)
    st.sidebar.success(f"Datos cargados: {len(df):,} registros")
else:
    df = generar_datos_simulados()
    st.sidebar.info(
        "Mostrando datos **simulados** de ejemplo.\n\n"
        "Carga un CSV real (con columnas como `fecha`, `localidad`, "
        "`tipo_accidente`, `gravedad`, `heridos`, `muertos`, `latitud`, "
        "`longitud`) para analizar información oficial, por ejemplo del "
        "portal de Datos Abiertos de Bogotá / SIMUR."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

años_disp = sorted(df["año"].dropna().unique().tolist())
años_sel = st.sidebar.multiselect("Año", años_disp, default=años_disp)

loc_disp = sorted(df["localidad"].dropna().unique().tolist())
loc_sel = st.sidebar.multiselect("Localidad", loc_disp, default=loc_disp)

tipo_disp = sorted(df["tipo_accidente"].dropna().unique().tolist())
tipo_sel = st.sidebar.multiselect("Tipo de accidente", tipo_disp, default=tipo_disp)

grav_disp = sorted(df["gravedad"].dropna().unique().tolist())
grav_sel = st.sidebar.multiselect("Gravedad", grav_disp, default=grav_disp)

df_f = df[
    df["año"].isin(años_sel)
    & df["localidad"].isin(loc_sel)
    & df["tipo_accidente"].isin(tipo_sel)
    & df["gravedad"].isin(grav_sel)
].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dashboard construido con Streamlit + Plotly. Todas las gráficas son "
    "interactivas: puedes hacer zoom, filtrar la leyenda y pasar el mouse "
    "sobre cada punto o barra para ver el detalle."
)

# --------------------------------------------------------------------------
# ENCABEZADO Y KPIs
# --------------------------------------------------------------------------
st.title("🚦 Estadísticas de Accidentes de Tránsito — Bogotá D.C.")
st.caption(
    "Explora la accidentalidad vial en la ciudad por localidad, tipo de "
    "accidente, gravedad, fecha y ubicación geográfica."
)

if df_f.empty:
    st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)
total_acc = len(df_f)
total_heridos = int(df_f["heridos"].sum())
total_muertos = int(df_f["muertos"].sum())
loc_top = df_f["localidad"].value_counts().idxmax()
tipo_top = df_f["tipo_accidente"].value_counts().idxmax()

col1.metric("Total accidentes", f"{total_acc:,}")
col2.metric("Personas heridas", f"{total_heridos:,}")
col3.metric("Personas fallecidas", f"{total_muertos:,}")
col4.metric("Localidad más afectada", loc_top)
col5.metric("Tipo más frecuente", tipo_top)

st.markdown("---")

# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📍 Por localidad", "📈 Tendencia temporal", "🚗 Tipo y gravedad",
     "🗺️ Mapa", "📋 Datos"]
)

# --- TAB 1: LOCALIDAD -------------------------------------------------
with tab1:
    st.subheader("Accidentes por localidad")
    conteo_loc = (
        df_f.groupby("localidad").size().reset_index(name="accidentes")
        .sort_values("accidentes", ascending=False)
    )
    fig_loc = px.bar(
        conteo_loc, x="accidentes", y="localidad", orientation="h",
        color="accidentes", color_continuous_scale="Reds",
        text="accidentes", title="Número de accidentes por localidad",
    )
    fig_loc.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    st.plotly_chart(fig_loc, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        grav_loc = df_f.groupby(["localidad", "gravedad"]).size().reset_index(name="conteo")
        fig_grav_loc = px.bar(
            grav_loc, x="localidad", y="conteo", color="gravedad",
            title="Gravedad de accidentes por localidad", barmode="stack",
        )
        fig_grav_loc.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig_grav_loc, use_container_width=True)
    with colB:
        heridos_muertos = (
            df_f.groupby("localidad")[["heridos", "muertos"]].sum()
            .reset_index().sort_values("heridos", ascending=False)
        )
        fig_hm = px.bar(
            heridos_muertos, x="localidad", y=["heridos", "muertos"],
            title="Heridos y fallecidos por localidad", barmode="group",
        )
        fig_hm.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig_hm, use_container_width=True)

# --- TAB 2: TENDENCIA TEMPORAL -----------------------------------------
with tab2:
    st.subheader("Evolución de los accidentes en el tiempo")

    serie_mensual = (
        df_f.assign(periodo=df_f["fecha"].dt.to_period("M").dt.to_timestamp())
        .groupby("periodo").size().reset_index(name="accidentes")
    )
    fig_ts = px.line(
        serie_mensual, x="periodo", y="accidentes", markers=True,
        title="Accidentes por mes",
    )
    fig_ts.update_layout(height=450)
    st.plotly_chart(fig_ts, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        if "dia_semana_nombre" in df_f.columns:
            orden_dias = DIAS_ES
            conteo_dia = df_f["dia_semana_nombre"].value_counts().reindex(orden_dias).reset_index()
            conteo_dia.columns = ["dia", "accidentes"]
            fig_dia = px.bar(
                conteo_dia, x="dia", y="accidentes", color="accidentes",
                color_continuous_scale="Blues", title="Accidentes por día de la semana",
            )
            st.plotly_chart(fig_dia, use_container_width=True)
    with colD:
        conteo_hora = df_f["hora"].value_counts().sort_index().reset_index()
        conteo_hora.columns = ["hora", "accidentes"]
        fig_hora = px.area(
            conteo_hora, x="hora", y="accidentes",
            title="Distribución de accidentes por hora del día",
        )
        st.plotly_chart(fig_hora, use_container_width=True)

    if {"dia_semana_nombre", "hora"}.issubset(df_f.columns):
        st.subheader("Mapa de calor: día de la semana vs. hora")
        heat = df_f.pivot_table(
            index="dia_semana_nombre", columns="hora", values="fecha",
            aggfunc="count", fill_value=0,
        ).reindex(DIAS_ES)
        fig_heat = px.imshow(
            heat, aspect="auto", color_continuous_scale="YlOrRd",
            labels=dict(x="Hora del día", y="Día", color="Accidentes"),
        )
        fig_heat.update_layout(height=450)
        st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 3: TIPO Y GRAVEDAD ---------------------------------------------
with tab3:
    st.subheader("Tipo de accidente, gravedad y causas")
    colE, colF = st.columns(2)
    with colE:
        fig_tipo = px.pie(
            df_f, names="tipo_accidente", title="Distribución por tipo de accidente",
            hole=0.4,
        )
        st.plotly_chart(fig_tipo, use_container_width=True)
    with colF:
        fig_grav = px.pie(
            df_f, names="gravedad", title="Distribución por gravedad", hole=0.4,
            color="gravedad",
            color_discrete_map={"Solo daños": "#4CAF50", "Con heridos": "#FF9800", "Con muertos": "#D32F2F"},
        )
        st.plotly_chart(fig_grav, use_container_width=True)

    if "clase_vehiculo" in df_f.columns:
        fig_veh = px.histogram(
            df_f, x="clase_vehiculo", color="gravedad",
            title="Accidentes por clase de vehículo y gravedad", barmode="stack",
        )
        fig_veh.update_layout(height=450)
        st.plotly_chart(fig_veh, use_container_width=True)

    if "causa_probable" in df_f.columns:
        conteo_causa = df_f["causa_probable"].value_counts().reset_index()
        conteo_causa.columns = ["causa", "accidentes"]
        fig_causa = px.bar(
            conteo_causa.sort_values("accidentes"), x="accidentes", y="causa",
            orientation="h", title="Causas probables más frecuentes",
            color="accidentes", color_continuous_scale="Purples",
        )
        fig_causa.update_layout(height=450)
        st.plotly_chart(fig_causa, use_container_width=True)

# --- TAB 4: MAPA ---------------------------------------------------------
with tab4:
    st.subheader("Ubicación geográfica de los accidentes")
    if {"latitud", "longitud"}.issubset(df_f.columns):
        muestra = df_f.sample(min(len(df_f), 4000), random_state=1)
        fig_map = px.scatter_mapbox(
            muestra, lat="latitud", lon="longitud", color="gravedad",
            hover_data=["localidad", "tipo_accidente", "fecha"] if "fecha" in muestra.columns else ["localidad", "tipo_accidente"],
            color_discrete_map={"Solo daños": "#4CAF50", "Con heridos": "#FF9800", "Con muertos": "#D32F2F"},
            zoom=10, height=650, title="Mapa de accidentes (muestra)",
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("El dataset no tiene columnas `latitud`/`longitud`, así que no se puede mostrar el mapa.")

# --- TAB 5: DATOS ---------------------------------------------------------
with tab5:
    st.subheader("Datos filtrados")
    st.dataframe(df_f, use_container_width=True, height=500)
    csv_out = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)", data=csv_out,
        file_name="accidentes_bogota_filtrado.csv", mime="text/csv",
    )

st.markdown("---")
st.caption(
    "⚠️ Si no cargaste un archivo propio, los datos mostrados son "
    "simulados con fines demostrativos y no representan cifras oficiales. "
    "Para análisis real, carga un CSV del portal de Datos Abiertos de "
    "Bogotá / SIMUR (Secretaría Distrital de Movilidad)."
)
