import os
import io
import contextlib
import heapq
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import folium
from folium.plugins import AntPath, Fullscreen, MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="FlightIQ — Predicción de Vuelos",
    page_icon="✈️",
    layout="wide",
)

# ============================================================
# ESTILOS (tema oscuro coherente, tipo app comercial)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; max-width: 1150px; }
#MainMenu, footer, header { visibility: hidden; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.45s ease both; }

.panel-header {
    background: radial-gradient(1200px 300px at 10% -10%, rgba(245,179,1,0.18), transparent),
                linear-gradient(135deg, #0B1220 0%, #131C33 55%, #1B2A4A 100%);
    padding: 34px 38px; border-radius: 18px; color: #F4F6FB; margin-bottom: 26px;
    border: 1px solid rgba(255,255,255,0.06);
}
.panel-header h1 { margin: 0; font-size: 30px; font-weight: 800; letter-spacing: -0.5px; }
.panel-header p { margin: 8px 0 0 0; font-size: 14px; color: #A9B3C7; }
.panel-header .badge {
    display: inline-block; margin-bottom: 14px; padding: 6px 14px; border-radius: 20px;
    background: rgba(245,179,1,0.14); color: #F5B301; font-size: 11px; font-weight: 700;
    letter-spacing: 0.6px; text-transform: uppercase; border: 1px solid rgba(245,179,1,0.25);
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: #121A2E; border-radius: 16px; border: 1px solid rgba(255,255,255,0.06);
}

.resultado-card {
    background: linear-gradient(180deg, #141D33 0%, #101828 100%);
    border-radius: 16px; border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 8px 24px rgba(0,0,0,0.28); padding: 22px 24px; margin-bottom: 16px; height: 100%;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.resultado-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 32px rgba(0,0,0,0.4);
}
.resultado-card.rapida { border-top: 3px solid #22C55E; }
.resultado-card.rapida:hover { border-color: rgba(34,197,94,0.5); }
.resultado-card.barata { border-top: 3px solid #EF4444; }
.resultado-card.barata:hover { border-color: rgba(239,68,68,0.5); }
.resultado-card.balanceada { border-top: 3px solid #3B82F6; }
.resultado-titulo { font-size: 16px; font-weight: 700; color: #F4F6FB; margin: 0 0 12px 0; }
.ruta-chip {
    display: inline-flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.06);
    border-radius: 20px; padding: 6px 14px; font-size: 12.5px; font-weight: 600;
    color: #C9D2E3; margin: 3px 4px 3px 0; border: 1px solid rgba(255,255,255,0.06);
    transition: background 0.15s ease;
}
.ruta-chip:hover { background: rgba(245,179,1,0.14); color: #F5B301; }
.metric-row { display: flex; gap: 22px; margin-top: 16px; flex-wrap: wrap; }
.metric .valor { font-size: 21px; font-weight: 800; color: #FFFFFF; }
.metric .etiqueta { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.5px; color: #7C879C; font-weight: 700; }

.map-title, .chart-title {
    font-size: 15px; font-weight: 700; color: #F4F6FB; margin: 8px 0 10px 2px;
}

.hist-item {
    background: #121A2E; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
    padding: 10px 14px; margin-bottom: 8px; font-size: 13px; color: #C9D2E3;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px;
    transition: background 0.15s ease, border-color 0.15s ease;
}
.hist-item:hover { background: #17203A; border-color: rgba(245,179,1,0.3); }
.hist-item .ts { color: #7C879C; font-size: 11.5px; }

/* Contenedor del mapa: leve resalte al pasar el mouse */
iframe { transition: filter 0.2s ease; border-radius: 12px; }
div[data-testid="stIFrame"]:hover iframe { filter: brightness(1.06); }

/* Pestañas más marcadas al pasar el mouse / al estar activas */
button[data-baseweb="tab"] { transition: color 0.15s ease; }
button[data-baseweb="tab"]:hover { color: #F5B301 !important; }

button[kind="primary"] {
    background: linear-gradient(135deg, #F5B301, #E09A00) !important;
    color: #101828 !important; font-weight: 700 !important; border: none !important;
    box-shadow: 0 6px 16px rgba(245,179,1,0.28) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(245,179,1,0.4) !important; }

.mapa-selector-hint {
    font-size: 12.5px; color: #7C879C; margin: 4px 0 10px 2px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="panel-header fade-in">
    <span class="badge">Sistema con IA · Machine Learning</span>
    <h1>✈️ FlightIQ — Panel de Predicción de Vuelos</h1>
    <p>Predicción de retrasos con Random Forest + optimización de rutas por tiempo y costo (Dijkstra)</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE DATOS, ENTRENAMIENTO DEL MODELO Y GRAFO
# (cacheado: se entrena una vez por sesión del servidor, no en cada clic)
# ============================================================
NOMBRE_ARCHIVO = "flight_delays_15_aerolineas.csv"
RUTA_CSV_LOCAL = os.path.join(os.path.dirname(__file__), "data", NOMBRE_ARCHIVO)


def obtener_ruta_dataset():
    """
    Busca el CSV en 2 lugares, en este orden:
    1. Un archivo local en data/ (si el repo lo incluye)
    2. Descarga desde Kaggle usando credenciales guardadas en Streamlit Secrets
    """
    if os.path.exists(RUTA_CSV_LOCAL):
        return RUTA_CSV_LOCAL

    try:
        if "KAGGLE_API_TOKEN" in st.secrets:
            os.environ["KAGGLE_API_TOKEN"] = st.secrets["KAGGLE_API_TOKEN"]
            import kagglehub
            carpeta_descarga = kagglehub.dataset_download("spmv1980/hackaton-2025-equipo-71")
            return os.path.join(carpeta_descarga, NOMBRE_ARCHIVO)
    except Exception as e:
        st.error(f"No se pudo descargar el dataset desde Kaggle: {e}")

    st.error(
        "⚠️ No encontré el archivo de datos. Agrega tu token de Kaggle en "
        "Settings → Secrets de Streamlit Cloud (KAGGLE_API_TOKEN), "
        "o incluye el CSV en data/ dentro del repositorio."
    )
    st.stop()


@st.cache_resource(show_spinner="🤖 Descargando datos y entrenando el modelo de IA...")
def preparar_sistema():
    columnas_necesarias = [
        'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
        'ORIGIN_AIRPORT_ID', 'ORIGIN',
        'DEST_AIRPORT_ID', 'DEST',
        'ARR_DELAY', 'DISTANCE'
    ]

    ruta_csv = obtener_ruta_dataset()
    df_sistema = pd.read_csv(ruta_csv, nrows=200000, usecols=columnas_necesarias).dropna().drop_duplicates()

    # Diccionarios de traducción
    mapa_local_aeropuertos = dict(zip(df_sistema['ORIGIN_AIRPORT_ID'], df_sistema['ORIGIN']))
    mapa_letras_a_id = {v: k for k, v in mapa_local_aeropuertos.items()}

    respaldo_manual = {
        "MIA": "Miami, FL", "JFK": "New York, NY", "LAX": "Los Angeles, CA",
        "ATL": "Atlanta, GA", "DFW": "Dallas/Fort Worth, TX", "ORD": "Chicago, IL",
        "DEN": "Denver, CO", "SEA": "Seattle, WA", "SFO": "San Francisco, CA",
        "SAN": "San Diego, CA", "BOS": "Boston, MA", "MSP": "Minneapolis, MN"
    }
    mapa_ciudades_comunes = dict(respaldo_manual)

    # Distancias reales por ruta
    mapa_distancias = {}
    for _, fila in df_sistema.iterrows():
        clave_ruta = (int(fila['ORIGIN_AIRPORT_ID']), int(fila['DEST_AIRPORT_ID']))
        mapa_distancias[clave_ruta] = float(fila['DISTANCE'])

    # Entrenamiento del modelo
    encoder_ori = LabelEncoder()
    encoder_des = LabelEncoder()

    df_ia = df_sistema.copy()
    df_ia['ORIGIN_AIRPORT_ID'] = encoder_ori.fit_transform(df_sistema['ORIGIN_AIRPORT_ID'])
    df_ia['DEST_AIRPORT_ID'] = encoder_des.fit_transform(df_sistema['DEST_AIRPORT_ID'])

    X = df_ia[['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']]
    y = df_ia['ARR_DELAY']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo_ia_vuelos = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    modelo_ia_vuelos.fit(X_train, y_train)

    # Grafo dirigido
    class GrafoSistema:
        def __init__(self):
            self.adjacencia = {}

        def agregar_arista(self, u, v, dist):
            self.adjacencia.setdefault(u, []).append({'destino': v, 'distancia': dist})

    grafo_vuelos = GrafoSistema()
    for (ori, des), distancia in mapa_distancias.items():
        grafo_vuelos.agregar_arista(ori, des, distancia)

    return {
        "mapa_local_aeropuertos": mapa_local_aeropuertos,
        "mapa_letras_a_id": mapa_letras_a_id,
        "mapa_ciudades_comunes": mapa_ciudades_comunes,
        "encoder_ori": encoder_ori,
        "encoder_des": encoder_des,
        "modelo_ia_vuelos": modelo_ia_vuelos,
        "grafo_vuelos": grafo_vuelos,
    }


@st.cache_resource(show_spinner="🌐 Descargando coordenadas de aeropuertos de EE.UU....")
def cargar_coordenadas():
    coordenadas = {
        "MIA": (-80.28, 25.79), "JFK": (-73.77, 40.64), "LAX": (-118.40, 33.94),
        "ATL": (-84.42, 33.64), "DFW": (-97.04, 32.89), "ORD": (-87.90, 41.97),
        "DEN": (-104.67, 39.85), "SEA": (-122.30, 47.45), "SFO": (-122.37, 37.62),
        "SAN": (-117.18, 32.73), "BOS": (-71.00, 42.36), "MSP": (-93.22, 44.88)
    }
    try:
        url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        for linea in resp.text.splitlines():
            campos = [c.strip('"') for c in linea.split(',')]
            if len(campos) < 8:
                continue
            if campos[3] != "United States" or not campos[4] or campos[4] == "\\N" or len(campos[4]) != 3:
                continue
            try:
                coordenadas[campos[4]] = (float(campos[7]), float(campos[6]))
            except ValueError:
                continue
    except Exception:
        pass
    return coordenadas


@st.cache_resource(show_spinner="🌐 Descargando nombres de ciudades de aeropuertos...")
def cargar_nombres_ciudades():
    """
    Arma código IATA -> nombre de ciudad para TODOS los aeropuertos (no solo los
    12 principales), así aeropuertos chicos como ABE o ATW también muestran su
    nombre real en vez del código de 3 letras.
    """
    nombres = {}
    try:
        url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        for linea in resp.text.splitlines():
            campos = [c.strip('"') for c in linea.split(',')]
            if len(campos) < 5:
                continue
            iata, ciudad = campos[4], campos[2]
            if not iata or iata == "\\N" or len(iata) != 3:
                continue
            if not ciudad or ciudad == "\\N":
                continue
            nombres[iata] = ciudad
    except Exception:
        pass
    return nombres


sistema = preparar_sistema()
coordenadas_reales = cargar_coordenadas()

mapa_local_aeropuertos = sistema["mapa_local_aeropuertos"]
mapa_letras_a_id = sistema["mapa_letras_a_id"]
# Los 12 nombres "manuales" (con estado, ej. "Miami, FL") tienen prioridad;
# para el resto de aeropuertos usamos el nombre real descargado.
mapa_ciudades_comunes = {**cargar_nombres_ciudades(), **sistema["mapa_ciudades_comunes"]}
encoder_ori = sistema["encoder_ori"]
encoder_des = sistema["encoder_des"]
modelo_ia_vuelos = sistema["modelo_ia_vuelos"]
grafo_vuelos = sistema["grafo_vuelos"]

mapa_ciudades_a_id = {}
for nodo_id, siglas in mapa_local_aeropuertos.items():
    nombre_ciudad = mapa_ciudades_comunes.get(siglas, siglas)
    mapa_ciudades_a_id[nombre_ciudad] = siglas
lista_ciudades_ordenadas = sorted(mapa_ciudades_a_id.keys())

# ============================================================
# FUNCIONES DE BÚSQUEDA (idénticas en espíritu a tu notebook)
# ============================================================
def a_horas(minutos):
    h = int(minutos // 60)
    m = int(minutos % 60)
    return f"{h}h {m}m" if h > 0 else f"{m}m"


def a_ciudad(nid):
    siglas = mapa_local_aeropuertos.get(nid, f"APT_{nid}")
    return mapa_ciudades_comunes.get(siglas, siglas)


@st.cache_data(show_spinner=False)
def predecir_retrasos_por_lote(mes, dia, dia_semana):
    filas_ia, claves_aristas = [], []
    for u in grafo_vuelos.adjacencia:
        for con in grafo_vuelos.adjacencia[u]:
            v = con['destino']
            filas_ia.append({
                'MONTH': mes, 'DAY_OF_MONTH': dia, 'DAY_OF_WEEK': dia_semana,
                'ORIGIN_AIRPORT_ID': u, 'DEST_AIRPORT_ID': v
            })
            claves_aristas.append((u, v))

    df_pred = pd.DataFrame(filas_ia)
    try:
        df_pred['ORIGIN_AIRPORT_ID'] = encoder_ori.transform(df_pred['ORIGIN_AIRPORT_ID'])
        df_pred['DEST_AIRPORT_ID'] = encoder_des.transform(df_pred['DEST_AIRPORT_ID'])
        retrasos = modelo_ia_vuelos.predict(df_pred)
    except Exception:
        retrasos = np.zeros(len(df_pred))

    return {(u, v): max(0, retrasos[i]) for i, (u, v) in enumerate(claves_aristas)}


# ============================================================
# MODELO ÚNICO Y CONSISTENTE DE TIEMPO/PRECIO POR TRAMO
# (antes cada ruta usaba su propia fórmula de precio, lo que podía
#  hacer que "económica" saliera más cara que "rápida". Ahora las
#  tres usan la MISMA fórmula; solo cambia qué está optimizando
#  el algoritmo de Dijkstra en cada caso).
# ============================================================
def tiempo_tramo(distancia, retraso):
    return (distancia / 500) * 60 + retraso


def precio_tramo(distancia, retraso, es_escala):
    cargo_escala = 55.0 if es_escala else 0.0
    return 42.0 + (distancia * 0.095) + (retraso * 1.10) + cargo_escala


def camino_rapido(ori_id, des_id, dict_retrasos):
    """Minimiza tiempo total. Por construcción, ninguna otra ruta puede tener menos tiempo que esta."""
    cola = [(0, 0, ori_id, [ori_id])]
    visitados = set()
    while cola:
        (t, c, nodo, cam) = heapq.heappop(cola)
        if nodo == des_id:
            return t, c, cam
        if nodo in visitados:
            continue
        visitados.add(nodo)
        for con in grafo_vuelos.adjacencia.get(nodo, []):
            sig = con['destino']
            if sig in visitados:
                continue
            ret = dict_retrasos.get((nodo, sig), 0)
            es_escala = nodo != ori_id
            tiempo = tiempo_tramo(con['distancia'], ret)
            precio = precio_tramo(con['distancia'], ret, es_escala)
            heapq.heappush(cola, (t + tiempo, c + precio, sig, cam + [sig]))
    return None, None, []


def camino_economico(ori_id, des_id, dict_retrasos):
    """Minimiza precio total. Por construcción, ninguna otra ruta puede costar menos que esta."""
    cola = [(0, 0, ori_id, [ori_id])]
    visitados = set()
    while cola:
        (c, t, nodo, cam) = heapq.heappop(cola)
        if nodo == des_id:
            return t, c, cam
        if nodo in visitados:
            continue
        visitados.add(nodo)
        for con in grafo_vuelos.adjacencia.get(nodo, []):
            sig = con['destino']
            if sig in visitados:
                continue
            ret = dict_retrasos.get((nodo, sig), 0)
            es_escala = nodo != ori_id
            tiempo = tiempo_tramo(con['distancia'], ret)
            precio = precio_tramo(con['distancia'], ret, es_escala)
            heapq.heappush(cola, (c + precio, t + tiempo, sig, cam + [sig]))
    return None, None, []


def retraso_promedio_ruta(ruta_ids, dict_retrasos):
    """Promedio de retraso predicho (minutos) en los tramos de una ruta."""
    if not ruta_ids or len(ruta_ids) < 2:
        return 0.0
    valores = [dict_retrasos.get((ruta_ids[i], ruta_ids[i + 1]), 0) for i in range(len(ruta_ids) - 1)]
    return float(np.mean(valores)) if valores else 0.0


def tarjeta_resultado_html(tipo, ruta_ids, tiempo_min, precio):
    config = {
        "rapida": ("🚀", "Ruta más rápida"),
        "barata": ("💵", "Ruta más económica"),
        "balanceada": ("⚖️", "Ruta balanceada"),
    }
    icono, titulo = config[tipo]
    duracion = a_horas(tiempo_min)
    escalas = max(0, len(ruta_ids) - 2)
    chips = ""
    for i, nid in enumerate(ruta_ids):
        ciudad = a_ciudad(nid)
        marcador = "✈️" if i == 0 else ("🏁" if i == len(ruta_ids) - 1 else "🔄")
        flecha = " → " if i < len(ruta_ids) - 1 else ""
        chips += f"<span class='ruta-chip'>{marcador} {ciudad}</span>{flecha}"
    return f"""
    <div class="resultado-card {tipo} fade-in">
        <p class="resultado-titulo">{icono} {titulo}</p>
        <div>{chips}</div>
        <div class="metric-row">
            <div class="metric"><div class="valor">{duracion}</div><div class="etiqueta">Duración</div></div>
            <div class="metric"><div class="valor">${precio:.2f}</div><div class="etiqueta">Precio estimado</div></div>
            <div class="metric"><div class="valor">{escalas}</div><div class="etiqueta">Escala(s)</div></div>
        </div>
    </div>
    """


def grafico_comparativo(rutas):
    """rutas: lista de (nombre, duracion_min, precio, color)"""
    fig = go.Figure()
    for nombre, dur, precio, color in rutas:
        fig.add_trace(go.Scatter(
            x=[dur], y=[precio], mode="markers+text", name=nombre,
            text=[nombre], textposition="top center",
            marker=dict(size=22, color=color, line=dict(width=2, color="#0B1220")),
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Duración (minutos)", yaxis_title="Precio (USD)",
        showlegend=False,
        font=dict(family="Inter, sans-serif", color="#C9D2E3"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    return fig


def gauge_retraso(minutos_retraso):
    if minutos_retraso < 15:
        color_barra, etiqueta = "#22C55E", "Bajo riesgo de retraso"
    elif minutos_retraso < 30:
        color_barra, etiqueta = "#F5B301", "Riesgo moderado de retraso"
    else:
        color_barra, etiqueta = "#EF4444", "Alto riesgo de retraso"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(minutos_retraso, 1),
        number={"suffix": " min", "font": {"color": "#F4F6FB", "size": 30}},
        gauge={
            "axis": {"range": [0, 60], "tickcolor": "#7C879C"},
            "bar": {"color": color_barra},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 15], "color": "rgba(34,197,94,0.15)"},
                {"range": [15, 30], "color": "rgba(245,179,1,0.15)"},
                {"range": [30, 60], "color": "rgba(239,68,68,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(l=20, r=20, t=30, b=10),
        font=dict(family="Inter, sans-serif", color="#C9D2E3"),
        annotations=[dict(text=etiqueta, x=0.5, y=-0.15, showarrow=False, font=dict(size=13, color=color_barra))],
    )
    return fig


# ============================================================
# SELECTOR EN EL MAPA (clic en un aeropuerto para elegirlo)
# ============================================================
with st.expander("🗺️ Elegir origen/destino haciendo clic en el mapa", expanded=False):
    st.markdown(
        '<p class="mapa-selector-hint">Primer clic = origen, segundo clic = destino. '
        'Un tercer clic reinicia la selección.</p>',
        unsafe_allow_html=True,
    )
    aeropuertos_clicables = sorted(set(mapa_ciudades_a_id.values()) & set(coordenadas_reales.keys()))

    mapa_selector = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB dark_matter")
    cluster = MarkerCluster().add_to(mapa_selector)
    for codigo in aeropuertos_clicables:
        ciudad = mapa_ciudades_comunes.get(codigo, codigo)
        folium.Marker(
            location=(coordenadas_reales[codigo][1], coordenadas_reales[codigo][0]),
            tooltip=f"{codigo} — {ciudad}",
            icon=folium.Icon(color="lightblue", icon="plane", prefix="fa"),
        ).add_to(cluster)

    estado_mapa = st_folium(
        mapa_selector, height=420, use_container_width=True,
        key="mapa_selector", returned_objects=["last_object_clicked_tooltip"],
    )

    click_actual = estado_mapa.get("last_object_clicked_tooltip") if estado_mapa else None
    if click_actual and click_actual != st.session_state.get("_ultimo_click_mapa"):
        st.session_state["_ultimo_click_mapa"] = click_actual
        codigo_click = click_actual.split(" — ")[0].strip()
        ciudad_click = mapa_ciudades_comunes.get(codigo_click, codigo_click)

        if not st.session_state.get("sel_origen"):
            st.session_state["sel_origen"] = ciudad_click
        elif not st.session_state.get("sel_destino") and ciudad_click != st.session_state.get("sel_origen"):
            st.session_state["sel_destino"] = ciudad_click
        else:
            st.session_state["sel_origen"] = ciudad_click
            st.session_state["sel_destino"] = None
        st.rerun()

# ============================================================
# FORMULARIO
# ============================================================
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        ciudad_ori = st.selectbox(
            "📍 Origen", options=lista_ciudades_ordenadas, index=None,
            placeholder="Elige una ciudad...", key="sel_origen",
        )
    with col2:
        ciudad_des = st.selectbox(
            "🏁 Destino", options=lista_ciudades_ordenadas, index=None,
            placeholder="Elige una ciudad...", key="sel_destino",
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        mes = st.slider("📅 Mes", 1, 12, 10)
    with col4:
        dia = st.slider("📆 Día", 1, 31, 15)
    with col5:
        dia_semana = st.slider("🕒 Día de la semana", 1, 7, 3)

    st.caption("⚡ Los resultados se actualizan solos apenas eliges origen, destino o cambias la fecha.")

# ============================================================
# BÚSQUEDA AUTOMÁTICA (sin botón — se recalcula solo cuando algo cambia)
# ============================================================
if "historial" not in st.session_state:
    st.session_state["historial"] = []

if not ciudad_ori or not ciudad_des:
    st.info("👆 Elige una ciudad de origen y una de destino para ver los resultados.")
else:
    clave_actual = (ciudad_ori, ciudad_des, mes, dia, dia_semana)

    if st.session_state.get("clave_calculada") != clave_actual:
        siglas_origen = mapa_ciudades_a_id.get(ciudad_ori)
        siglas_destino = mapa_ciudades_a_id.get(ciudad_des)
        ori_id = mapa_letras_a_id.get(siglas_origen)
        des_id = mapa_letras_a_id.get(siglas_destino)

        with st.spinner("🤖 Calculando retrasos con IA y optimizando rutas..."):
            dict_retrasos = predecir_retrasos_por_lote(mes, dia, dia_semana)
            t_rapido, c_rapido, ruta_rapida = camino_rapido(ori_id, des_id, dict_retrasos)
            t_barato, c_barato, ruta_barata = camino_economico(ori_id, des_id, dict_retrasos)
            retraso_prom = retraso_promedio_ruta(ruta_rapida, dict_retrasos) if ruta_rapida else 0.0

        st.session_state["resultado"] = {
            "ruta_rapida": ruta_rapida, "t_rapido": t_rapido, "c_rapido": c_rapido,
            "ruta_barata": ruta_barata, "t_barato": t_barato, "c_barato": c_barato,
            "retraso_prom": retraso_prom,
        }
        st.session_state["clave_calculada"] = clave_actual

        if ruta_rapida:
            st.session_state["historial"].insert(0, {
                "hora": datetime.now().strftime("%H:%M:%S"),
                "origen": ciudad_ori, "destino": ciudad_des,
                "fecha_vuelo": f"{mes}/{dia} (día sem. {dia_semana})",
                "duracion": a_horas(t_rapido), "precio": f"${c_rapido:.2f}",
            })
            st.session_state["historial"] = st.session_state["historial"][:8]

# ============================================================
# RESULTADOS
# ============================================================
resultado = st.session_state.get("resultado")
if resultado and ciudad_ori and ciudad_des:
    ruta_rapida = resultado["ruta_rapida"]
    ruta_barata = resultado["ruta_barata"]

    if not ruta_rapida:
        st.warning("❌ No se encontró una ruta viable entre estas dos ciudades.")
    else:
        colA, colB = st.columns(2)
        with colA:
            st.markdown(tarjeta_resultado_html("rapida", ruta_rapida, resultado["t_rapido"], resultado["c_rapido"]), unsafe_allow_html=True)
        with colB:
            st.markdown(tarjeta_resultado_html("barata", ruta_barata, resultado["t_barato"], resultado["c_barato"]), unsafe_allow_html=True)

        tab_mapa, tab_grafico, tab_tabla = st.tabs(["🗺️ Mapa", "📊 Gráfico", "📋 Tabla"])

        # --- TAB: MAPA ---
        with tab_mapa:
            nombres_rapida = [mapa_local_aeropuertos.get(nid) for nid in ruta_rapida]
            nombres_barata = [mapa_local_aeropuertos.get(nid) for nid in ruta_barata]

            def con_coordenadas(lista):
                return [s for s in lista if s in coordenadas_reales]

            validos_rapida = con_coordenadas(nombres_rapida)
            validos_barata = con_coordenadas(nombres_barata)
            faltantes = set([s for s in (nombres_rapida + nombres_barata) if s and s not in coordenadas_reales])

            if faltantes:
                st.caption(f"⚠️ No tengo coordenadas guardadas para: {', '.join(sorted(faltantes))}. Esos tramos no se dibujan en el mapa.")

            activos = list(set(validos_rapida + validos_barata))
            if len(activos) < 2:
                st.info("No hay suficientes aeropuertos con coordenadas conocidas para dibujar el mapa.")
            else:
                coords = [coordenadas_reales[s] for s in activos]
                centro_lat = sum(c[1] for c in coords) / len(coords)
                centro_lon = sum(c[0] for c in coords) / len(coords)

                m = folium.Map(location=[centro_lat, centro_lon], zoom_start=5, tiles="CartoDB dark_matter")
                Fullscreen(position="topright").add_to(m)

                mismas_rutas = (validos_rapida == validos_barata) and len(validos_rapida) > 1
                OFFSET = 0.35

                def dibujar_ruta(lista_siglas, color, nombre_ruta, desplazar=False):
                    puntos = [(coordenadas_reales[s][1], coordenadas_reales[s][0]) for s in lista_siglas]
                    if desplazar:
                        puntos = [(lat + OFFSET, lon + OFFSET) for lat, lon in puntos]
                    AntPath(puntos, color=color, weight=4, opacity=0.9, delay=800, dash_array=[12, 18], tooltip=nombre_ruta).add_to(m)
                    for i, s in enumerate(lista_siglas):
                        ciudad = mapa_ciudades_comunes.get(s, s)
                        if i == 0:
                            icono, etiqueta, color_marcador = "plane-departure", f"Origen: {ciudad} ({s})", "green"
                        elif i == len(lista_siglas) - 1:
                            icono, etiqueta, color_marcador = "flag-checkered", f"Destino: {ciudad} ({s})", "red"
                        else:
                            icono, etiqueta, color_marcador = "arrows-turn-right", f"Escala: {ciudad} ({s})", "orange"
                        folium.Marker(
                            location=(coordenadas_reales[s][1], coordenadas_reales[s][0]),
                            popup=etiqueta, tooltip=etiqueta,
                            icon=folium.Icon(color=color_marcador, icon=icono, prefix='fa')
                        ).add_to(m)

                if len(validos_rapida) > 1:
                    dibujar_ruta(validos_rapida, "#22C55E", "🚀 Más rápida")
                if len(validos_barata) > 1:
                    dibujar_ruta(validos_barata, "#3B82F6", "💵 Más económica", desplazar=mismas_rutas)

                if mismas_rutas:
                    st.caption("ℹ️ La ruta más rápida y la más económica pasan por los mismos aeropuertos — se separaron un poco visualmente para que se vean ambas líneas.")

                st_folium(m, width=None, height=480, use_container_width=True, key="mapa_resultado")

        # --- TAB: GRÁFICO ---
        with tab_grafico:
            colChart, colGauge = st.columns([2, 1])
            with colChart:
                st.markdown('<p class="chart-title">📊 Comparación de rutas (duración vs. precio)</p>', unsafe_allow_html=True)
                rutas_chart = [
                    ("🚀 Rápida", resultado["t_rapido"], resultado["c_rapido"], "#22C55E"),
                    ("💵 Económica", resultado["t_barato"], resultado["c_barato"], "#3B82F6"),
                ]
                st.plotly_chart(grafico_comparativo(rutas_chart), use_container_width=True, config={"displayModeBar": False})
            with colGauge:
                st.markdown('<p class="chart-title">🚦 Probabilidad de retraso (ruta rápida)</p>', unsafe_allow_html=True)
                st.plotly_chart(gauge_retraso(resultado["retraso_prom"]), use_container_width=True, config={"displayModeBar": False})

        # --- TAB: TABLA ---
        with tab_tabla:
            df_comparacion = pd.DataFrame({
                "Métrica": ["Duración", "Precio estimado", "Escala(s)", "Retraso promedio predicho"],
                "🚀 Ruta rápida": [
                    a_horas(resultado["t_rapido"]), f"${resultado['c_rapido']:.2f}",
                    str(max(0, len(ruta_rapida) - 2)), f"{resultado['retraso_prom']:.1f} min",
                ],
                "💵 Ruta económica": [
                    a_horas(resultado["t_barato"]), f"${resultado['c_barato']:.2f}",
                    str(max(0, len(ruta_barata) - 2)), "—",
                ],
            })
            st.dataframe(df_comparacion, use_container_width=True, hide_index=True)

# ============================================================
# HISTORIAL DE BÚSQUEDAS
# ============================================================
if st.session_state["historial"]:
    with st.expander(f"🕘 Historial de búsquedas recientes ({len(st.session_state['historial'])})"):
        for item in st.session_state["historial"]:
            st.markdown(f"""
            <div class="hist-item">
                <span>📍 <b>{item['origen']}</b> → 🏁 <b>{item['destino']}</b> · {item['fecha_vuelo']}</span>
                <span>⏱️ {item['duracion']} &nbsp;|&nbsp; 💰 {item['precio']} <span class="ts">({item['hora']})</span></span>
            </div>
            """, unsafe_allow_html=True)
