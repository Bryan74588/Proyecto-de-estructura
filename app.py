import os
import io
import contextlib
import heapq

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import st_folium

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="FlightIQ — Predicción de Vuelos",
    page_icon="✈️",
    layout="wide",
)

# ============================================================
# ESTILOS (look de página web profesional)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.block-container { padding-top: 1.2rem; max-width: 1100px; }

.panel-header {
    background: linear-gradient(135deg, #0B1E3D 0%, #1B3A6B 55%, #2E4374 100%);
    padding: 32px 36px; border-radius: 16px; color: white; margin-bottom: 24px;
}
.panel-header h1 { margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; }
.panel-header p { margin: 6px 0 0 0; font-size: 14px; opacity: 0.85; }
.panel-header .badge {
    display: inline-block; margin-bottom: 12px; padding: 5px 12px; border-radius: 20px;
    background: rgba(255,255,255,0.12); font-size: 11px; font-weight: 600;
    letter-spacing: 0.5px; text-transform: uppercase;
}

.resultado-card {
    background: white; border-radius: 14px; border: 1px solid #E9ECF2;
    box-shadow: 0 2px 10px rgba(15,30,60,0.05); padding: 20px 24px; margin-bottom: 16px;
}
.resultado-card.rapida { border-left: 5px solid #16A34A; }
.resultado-card.barata { border-left: 5px solid #DC2626; }
.resultado-titulo { font-size: 16px; font-weight: 700; color: #1B2A4A; margin: 0 0 12px 0; }
.ruta-chip {
    display: inline-flex; align-items: center; gap: 6px; background: #F1F5F9;
    border-radius: 20px; padding: 6px 14px; font-size: 13px; font-weight: 600;
    color: #334155; margin: 3px 4px 3px 0;
}
.metric-row { display: flex; gap: 28px; margin-top: 14px; flex-wrap: wrap; }
.metric .valor { font-size: 20px; font-weight: 800; color: #0F172A; }
.metric .etiqueta { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #64748B; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="panel-header">
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
        if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
            import kagglehub
            carpeta_descarga = kagglehub.dataset_download("spmv1980/hackaton-2025-equipo-71")
            return os.path.join(carpeta_descarga, NOMBRE_ARCHIVO)
    except Exception as e:
        st.error(f"No se pudo descargar el dataset desde Kaggle: {e}")

    st.error(
        "⚠️ No encontré el archivo de datos. Agrega tus credenciales de Kaggle en "
        "Settings → Secrets de Streamlit Cloud (KAGGLE_USERNAME y KAGGLE_KEY), "
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


sistema = preparar_sistema()
coordenadas_reales = cargar_coordenadas()

mapa_local_aeropuertos = sistema["mapa_local_aeropuertos"]
mapa_letras_a_id = sistema["mapa_letras_a_id"]
mapa_ciudades_comunes = sistema["mapa_ciudades_comunes"]
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


def camino_rapido(ori_id, des_id, dict_retrasos):
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
            precio = 40.0 + (con['distancia'] * 0.11) + (ret * 1.50)
            tiempo = (con['distancia'] / 500) * 60 + ret
            heapq.heappush(cola, (t + tiempo, c + precio, sig, cam + [sig]))
    return None, None, []


def camino_economico(ori_id, des_id, dict_retrasos):
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
            cargo_escala = 60.0 if nodo != ori_id else 0.0
            precio = 35.0 + (con['distancia'] * 0.08) + (ret * 0.80) + cargo_escala
            tiempo = (con['distancia'] / 500) * 60 + ret
            heapq.heappush(cola, (c + precio, t + tiempo, sig, cam + [sig]))
    return None, None, []


def tarjeta_resultado_html(tipo, ruta_ids, tiempo_min, precio):
    icono = "🚀" if tipo == "rapida" else "💵"
    titulo = "Ruta más rápida" if tipo == "rapida" else "Ruta más económica"
    duracion = a_horas(tiempo_min)
    escalas = max(0, len(ruta_ids) - 2)
    chips = ""
    for i, nid in enumerate(ruta_ids):
        ciudad = a_ciudad(nid)
        marcador = "✈️" if i == 0 else ("🏁" if i == len(ruta_ids) - 1 else "🔄")
        flecha = " → " if i < len(ruta_ids) - 1 else ""
        chips += f"<span class='ruta-chip'>{marcador} {ciudad}</span>{flecha}"
    return f"""
    <div class="resultado-card {tipo}">
        <p class="resultado-titulo">{icono} {titulo}</p>
        <div>{chips}</div>
        <div class="metric-row">
            <div class="metric"><div class="valor">{duracion}</div><div class="etiqueta">Duración</div></div>
            <div class="metric"><div class="valor">${precio:.2f}</div><div class="etiqueta">Precio estimado</div></div>
            <div class="metric"><div class="valor">{escalas}</div><div class="etiqueta">Escala(s)</div></div>
        </div>
    </div>
    """


# ============================================================
# FORMULARIO
# ============================================================
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        ciudad_ori = st.selectbox("📍 Origen", options=lista_ciudades_ordenadas, index=None, placeholder="Elige una ciudad...")
    with col2:
        ciudad_des = st.selectbox("🏁 Destino", options=lista_ciudades_ordenadas, index=None, placeholder="Elige una ciudad...")

    col3, col4, col5 = st.columns(3)
    with col3:
        mes = st.slider("📅 Mes", 1, 12, 10)
    with col4:
        dia = st.slider("📆 Día", 1, 31, 15)
    with col5:
        dia_semana = st.slider("🕒 Día de la semana", 1, 7, 3)

    buscar = st.button("Buscar y Comparar Vuelos", type="primary", use_container_width=False)

# ============================================================
# RESULTADOS
# ============================================================
if buscar:
    if not ciudad_ori or not ciudad_des:
        st.error("❌ Selecciona una ciudad de origen y una de destino.")
    else:
        siglas_origen = mapa_ciudades_a_id.get(ciudad_ori)
        siglas_destino = mapa_ciudades_a_id.get(ciudad_des)
        ori_id = mapa_letras_a_id.get(siglas_origen)
        des_id = mapa_letras_a_id.get(siglas_destino)

        with st.spinner("🤖 Calculando retrasos con IA y optimizando rutas..."):
            dict_retrasos = predecir_retrasos_por_lote(mes, dia, dia_semana)
            t_rapido, c_rapido, ruta_rapida = camino_rapido(ori_id, des_id, dict_retrasos)
            t_barato, c_barato, ruta_barata = camino_economico(ori_id, des_id, dict_retrasos)

        if not ruta_rapida:
            st.warning("❌ No se encontró una ruta viable entre estas dos ciudades.")
        else:
            colA, colB = st.columns(2)
            with colA:
                st.markdown(tarjeta_resultado_html("rapida", ruta_rapida, t_rapido, c_rapido), unsafe_allow_html=True)
            with colB:
                st.markdown(tarjeta_resultado_html("barata", ruta_barata, t_barato, c_barato), unsafe_allow_html=True)

            # --- MAPA INTERACTIVO ---
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

                m = folium.Map(location=[centro_lat, centro_lon], zoom_start=5, tiles="CartoDB positron")

                def dibujar_ruta(lista_siglas, color, nombre_ruta):
                    puntos = [(coordenadas_reales[s][1], coordenadas_reales[s][0]) for s in lista_siglas]
                    folium.PolyLine(puntos, color=color, weight=4, opacity=0.85, tooltip=nombre_ruta).add_to(m)
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
                    dibujar_ruta(validos_rapida, "#16A34A", "🚀 Más rápida")
                if len(validos_barata) > 1:
                    dibujar_ruta(validos_barata, "#DC2626", "💵 Más económica")

                st.markdown("#### 🗺️ Mapa de la ruta — escalas y destino")
                st_folium(m, width=None, height=480, use_container_width=True)
