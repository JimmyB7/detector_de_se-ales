"""
Sistema Inteligente de Control Vehicular y Detector de Señales de Tránsito
Dashboard Streamlit (Dark Glassmorphism / Cyberpunk) - Semanas 02-05, 07-11 y 13-16

Instalar:  pip install streamlit plotly pandas numpy pillow networkx scikit-learn scikit-image matplotlib pyvis
Ejecutar:  streamlit run src/app_gui.py
"""
import heapq, io, pickle, sqlite3, sys, time
from contextlib import closing
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
except ImportError:
    TfidfVectorizer = cosine_similarity = train_test_split = make_pipeline = None
    StandardScaler = LogisticRegression = RandomForestClassifier = MLPClassifier = None
    accuracy_score = confusion_matrix = precision_score = recall_score = None
try:
    from skimage.feature import canny
    from skimage.feature import local_binary_pattern
    from skimage.filters import threshold_otsu
    from skimage.measure import label as sk_label
    from skimage.color import label2rgb
except ImportError:
    canny = None
    local_binary_pattern = None
try:
    from pyvis.network import Network
except ImportError:
    Network = None

# Importar motor de inferencia de la semana 13
try:
    from semana13_motor_inferencia_transito import inferir_transito, reglas_senales
except ImportError:
    inferir_transito = reglas_senales = None
try:
    from semana15_conocimiento_vial import (
        buscar_articulos_validos,
        cargar_datos_iniciales,
        conectar,
        inicializar_base_conocimiento,
        validar_articulo_por_id,
    )
except ImportError:
    buscar_articulos_validos = cargar_datos_iniciales = conectar = None
    inicializar_base_conocimiento = validar_articulo_por_id = None
try:
    from semana16_detector_transito_pruebas import REFERENCIAS_VIALES, motor_de_referencia
except ImportError:
    REFERENCIAS_VIALES = motor_de_referencia = None

# ---------------------------------------------------------------- Rutas
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "src" else HERE
ARTIFACTS, UPLOADS, REPORTS = ROOT / "artifacts", ROOT / "uploads", ROOT / "reports"
for p in (ARTIFACTS, UPLOADS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

st.set_page_config(page_title="Control Vehicular IA", page_icon="🚦", layout="wide")

CLASES = {0: "Pare", 1: "Ceda el Paso", 2: "Limite Velocidad", 3: "Prohibido Girar"}
ONTO = {0: "pare", 1: "ceda_el_paso", 2: "limite_velocidad", 3: "prohibido_girar"}
REGLAS = {  # señal -> (acción, prioridad, color)
    "Pare": ("Detención total inmediata", "Crítico", "#ff2e63"),
    "Ceda el Paso": ("Reducir y ceder la vía", "Medio", "#ffb300"),
    "Limite Velocidad": ("Ajustar velocidad al límite", "Medio", "#ffb300"),
    "Prohibido Girar": ("Mantener carril y trayectoria", "Bajo", "#08fdd8"),
}

# ---------------------------------------------------------------- Estilos
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 15% 10%,#1b1038 0,#0a0f24 45%,#050816 100%);color:#e6f1ff}
[data-testid=stSidebar]{background:rgba(10,15,36,.85);border-right:1px solid #08fdd855}
h1,h2,h3{color:#08fdd8!important;text-shadow:0 0 12px #08fdd855}
.glass{background:rgba(255,255,255,.05);backdrop-filter:blur(12px);border:1px solid rgba(8,253,216,.25);
 border-radius:16px;padding:16px 20px;box-shadow:0 0 22px rgba(124,58,237,.25);transition:.3s;animation:fade .6s}
.glass:hover{transform:translateY(-3px);box-shadow:0 0 28px rgba(8,253,216,.45)}
.kpi b{display:block;font-size:2rem;color:#08fdd8}.kpi span{color:#9fb3d9;font-size:.85rem}
.pill{display:inline-block;padding:2px 12px;margin:2px;border-radius:999px;font-size:.78rem;font-weight:600;color:#050816}
.chat-u{background:#7c3aed55;border-radius:14px;padding:10px 14px;margin:6px 0;text-align:right}
.chat-a{background:#08fdd822;border:1px solid #08fdd855;border-radius:14px;padding:10px 14px;margin:6px 0}
@keyframes fade{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.stButton>button{background:linear-gradient(90deg,#7c3aed,#08fdd8);color:#050816;font-weight:700;border:0;border-radius:10px}
</style>""", unsafe_allow_html=True)


def kpi(label, value):
    st.markdown(f'<div class="glass kpi"><b>{value}</b><span>{label}</span></div>', unsafe_allow_html=True)


def pill(text, color):
    return f'<span class="pill" style="background:{color}">{text}</span>'


def timed(label, t0):
    ms = (time.perf_counter() - t0) * 1000
    st.caption(f"⏱️ {label}: {ms:.1f} ms")


PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

# ---------------------------------------------------------------- Lógica reutilizada
@st.cache_resource
def cargar_modelo():
    ruta = ARTIFACTS / "modelo_mlp.pkl"
    if not ruta.is_file():
        return None, f"No se encontró el modelo requerido: {ruta.relative_to(ROOT)}."
    try:
        with open(ruta, "rb") as f:
            modelo = pickle.load(f)
    except (OSError, EOFError, pickle.PickleError, ImportError, AttributeError, ValueError) as error:
        return None, f"No se pudo cargar el modelo: {error}"

    if isinstance(modelo, dict):
        estimador = modelo.get("modelo")
        dimensiones = getattr(estimador, "n_features_in_", None)
        clases = getattr(estimador, "classes_", None)
        return None, (
            "El checkpoint no es compatible con el detector: "
            f"el estimador interno declara {dimensiones} características y clases {clases}; "
            "se requieren 1024 características y las clases 0, 1, 2 y 3."
        )

    dimensiones = getattr(modelo, "n_features_in_", None)
    clases = getattr(modelo, "classes_", None)
    if not callable(getattr(modelo, "predict", None)):
        return None, "El archivo no contiene un clasificador compatible."
    if dimensiones != 32 * 32 or clases is None:
        return None, "El modelo debe aceptar 1024 características y declarar las cuatro clases de señales."

    try:
        clases_modelo = {int(clase) for clase in clases}
    except (TypeError, ValueError):
        return None, "El modelo contiene etiquetas de clase no válidas."
    if clases_modelo != {0, 1, 2, 3}:
        return None, (
            "El modelo no coincide con las cuatro clases del detector: "
            f"se encontraron {sorted(clases_modelo)}."
        )
    return modelo, None


def clasificar_imagen(img: Image.Image, mlp):
    if mlp is None:
        raise ValueError("No hay un modelo compatible para clasificar la imagen.")
    g32 = np.array(img.convert("L").resize((32, 32)), dtype=np.float32) / 255.0
    prediccion = int(mlp.predict(g32.reshape(1, -1))[0])
    if prediccion not in CLASES:
        raise ValueError(f"El modelo devolvió una clase desconocida: {prediccion}.")
    return prediccion


MODELO_MLP, ERROR_MODELO_MLP = cargar_modelo()


def guardar_sqlite(pid):
    with sqlite3.connect(ARTIFACTS / "imagenes.db") as con:
        crear_tabla_registros(con)
        con.execute("INSERT INTO imagenes_senales (etiqueta_id,etiqueta_nombre,split) VALUES (?,?,?)",
                    (pid, CLASES[pid], "subida_usuario"))


def crear_tabla_registros(con):
    con.execute(
        "CREATE TABLE IF NOT EXISTS imagenes_senales ("
        "id INTEGER PRIMARY KEY, etiqueta_id INTEGER, etiqueta_nombre TEXT, split TEXT)"
    )


def leer_sqlite(n=15):
    if not isinstance(n, int) or n < 1:
        raise ValueError("El límite de registros debe ser un entero positivo.")
    with sqlite3.connect(ARTIFACTS / "imagenes.db") as con:
        crear_tabla_registros(con)
        return pd.read_sql_query(
            "SELECT * FROM imagenes_senales ORDER BY id DESC LIMIT ?",
            con,
            params=(n,),
        )


def actualizar_ontologia(pid):
    ruta = ARTIFACTS / "ontologia.graphml"
    G = nx.read_graphml(ruta) if ruta.exists() else nx.DiGraph()
    G.add_edge("imagen_usuario", "prediccion", rel="genera")
    G.add_edge("prediccion", ONTO[pid], rel="asigna_clase")
    nx.write_graphml(G, ruta)
    return G


def cargar_ontologia():
    ruta = ARTIFACTS / "ontologia.graphml"
    if ruta.exists():
        return nx.read_graphml(ruta)
    G = nx.DiGraph()
    for n in ONTO.values():
        G.add_edge("senal_transito", n, rel="tiene_clase")
    return G


def vision(img: Image.Image, sigma: float):
    gris = np.array(img.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    bordes = canny(gris, sigma=sigma)
    umbral = threshold_otsu(gris)
    mascara = gris > umbral
    etiquetas, n = sk_label(mascara, return_num=True)
    color = label2rgb(etiquetas, image=gris, bg_label=0)
    return gris, bordes, mascara, color, umbral, n


# ---------------------------------------------------------------- A* y Minimax
def astar(grid, ini, fin):
    h = lambda a: abs(a[0] - fin[0]) + abs(a[1] - fin[1])
    abierta, prev, g, orden = [(h(ini), ini)], {}, {ini: 0}, []
    while abierta:
        _, cur = heapq.heappop(abierta)
        if cur in orden:
            continue
        orden.append(cur)
        if cur == fin:
            ruta = [cur]
            while cur in prev:
                cur = prev[cur]
                ruta.append(cur)
            return orden, ruta[::-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cur[0] + dx, cur[1] + dy)
            if 0 <= nb[0] < len(grid) and 0 <= nb[1] < len(grid) and grid[nb[0]][nb[1]] == 0:
                ng = g[cur] + 1
                if ng < g.get(nb, 1e9):
                    g[nb], prev[nb] = ng, cur
                    heapq.heappush(abierta, (ng + h(nb), nb))
    return orden, []


MANIOBRAS = ["Frenar", "Mantener", "Cambiar carril"]
RIVAL = ["Peatón cruza", "Peatón espera", "Vehículo invade"]


def minimax(payoff):
    """Nosotros maximizamos; el entorno (adversario) minimiza."""
    peor = payoff.min(axis=1)
    return int(peor.argmax()), peor


# ---------------------------------------------------------------- Datos de las prácticas
CASOS = [
    {"id": "C-001", "titulo": "Visión por computador y reglas", "desc": "Sistema de visión por computador para detectar señales de PARE e inferir la acción del vehículo."},
    {"id": "C-002", "titulo": "Red neuronal y control autónomo", "desc": "Monitoreo de carril e identificación de límites de velocidad mediante red neuronal."},
    {"id": "C-003", "titulo": "Decisión ética y priorización", "desc": "Módulo de decisión ética y priorización de frenado autónomo ante obstáculos imprevistos."},
]
KB_PATH = ROOT / "data" / "base_conocimiento.txt"
if not KB_PATH.is_file():
    raise FileNotFoundError(f"No se encontró la base de conocimiento: {KB_PATH}")
NORMAS = [line.strip() for line in KB_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
if not NORMAS:
    raise ValueError(f"La base de conocimiento está vacía: {KB_PATH}")
REGLAS_KW = {"R-PARE": ["pare", "detener"], "R-VEL": ["velocidad", "km/h"], "R-GIRO": ["girar", "retorno"],
             "R-PRIORIDAD": ["ceda", "prioridad", "rotonda"], "R-PEATON": ["peaton", "peatón", "peatones"]}


@st.cache_resource
def indexar_normas():
    if TfidfVectorizer is None:
        return None, None
    vec = TfidfVectorizer(lowercase=True)
    return vec, vec.fit_transform(NORMAS)


def buscar_normas(q, k=3):
    vec, mat = indexar_normas()
    if vec is None:
        raise RuntimeError("Instala scikit-learn: pip install scikit-learn")
    sims = cosine_similarity(vec.transform([q]), mat).ravel()
    out = []
    for i in sims.argsort()[::-1][:k]:
        txt = NORMAS[i].lower()
        disparadas = [r for r, kws in REGLAS_KW.items() if any(w in txt and w in q.lower() for w in kws)]
        out.append({"norma_id": int(i), "score_tfidf": round(float(sims[i]), 4), "reglas_disparadas": disparadas,
                    "evidencia": NORMAS[i]})
    return out


# ---------------------------------------------------------------- Sidebar
st.sidebar.title("🚦 Control Vehicular IA")
st.sidebar.caption("Dashboard del semestre · Semanas 02-05, 07-11 y 13-16")
MODULOS = [
    "🏠 Sistema Integrado", 
    "S02 · Clasificación y Reglas", 
    "S03 · Taxonomía de Casos",
    "S04 · A* y Minimax", 
    "S05 · Híbrido TF-IDF",
    "S07 · Representación y autómata", 
    "S08 · MLP, SQLite y Ontología",
    "S09 · Visión Artificial", 
    "S10 · Descriptores LBP", 
    "S11 · Benchmark de Modelos",
    "S13 · Motor de Inferencia Lógica",
    "S14 · Red Semántica de Tránsito",
    "S15 · Base de Conocimiento Vial",
    "S16 · Trazabilidad de Referencias"
]
pagina = st.sidebar.radio("Navegación", MODULOS)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Modelo MLP: {'✅ compatible (4 clases)' if MODELO_MLP is not None else '⚠️ no compatible'}"
)
st.sidebar.caption(f"scikit-image: {'✅' if canny else '❌'} · pyvis: {'✅' if Network else '❌ (usa Plotly)'}")

# ================================================================ PÁGINAS
def pag_s02():
    st.header("Semana 02 · Clasificación visual y motor de reglas")
    if model_dependencies_missing():
        st.error("Instala scikit-learn para ejecutar la práctica de la semana 02.")
        return
    from semana02_fundamentos import X_test, model, y_test

    predicciones = model.predict(X_test)
    cm = confusion_matrix(y_test, predicciones, labels=[0, 1, 2])
    acc = accuracy_score(y_test, predicciones)
    prec = precision_score(y_test, predicciones, labels=[0, 1, 2], average="macro", zero_division=0)
    rec = recall_score(y_test, predicciones, labels=[0, 1, 2], average="macro", zero_division=0)
    c1, c2, c3 = st.columns(3)
    with c1: kpi("Accuracy", f"{acc:.1%}")
    with c2: kpi("Precisión (macro)", f"{prec:.1%}")
    with c3: kpi("Recuerdo (macro)", f"{rec:.1%}")
    nombres = list(CLASES.values())[:3]
    fig = px.imshow(cm, x=nombres, y=nombres, text_auto=True, color_continuous_scale="Tealgrn",
                    labels=dict(x="Predicho", y="Real"))
    fig.update_layout(title="Matriz de confusión (modelo de la semana 02)", **PLOT)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Simulación de fotogramas + motor de reglas")
    frames = st.multiselect("Secuencia de fotogramas detectados", nombres, default=["Limite Velocidad", "Ceda el Paso", "Pare"])
    cols = st.columns(max(len(frames), 1))
    for i, (col, s) in enumerate(zip(cols, frames)):
        accion, prio, color = REGLAS[s]
        col.markdown(f'<div class="glass"><small>Fotograma {i+1}</small><h4>{s}</h4>{pill(prio, color)}'
                     f'<p>{accion}</p></div>', unsafe_allow_html=True)
    if any(REGLAS[s][1] == "Crítico" for s in frames):
        st.toast("Prioridad CRÍTICA detectada: frenado inmediato", icon="🚨")


def model_dependencies_missing():
    return any(dependency is None for dependency in (
        TfidfVectorizer, accuracy_score, confusion_matrix, precision_score, recall_score
    ))


def pag_s03():
    st.header("Semana 03 · Análisis taxonómico de casos viales")
    if model_dependencies_missing():
        st.error("Instala scikit-learn para clasificar los casos por área de inteligencia artificial.")
        return
    from semana03_taxonomia import classify_problem

    casos_clasificados = []
    for caso in CASOS:
        principal, areas, _ = classify_problem(caso["desc"])
        casos_clasificados.append({**caso, "cat": principal, "areas": areas})
    colores = {
        "Visión Artificial": "#08fdd8",
        "Sistemas Basados en Reglas": "#ffb300",
        "Robótica y Control": "#ff2e63",
        "General": "#7c3aed",
    }
    q = st.text_input("🔎 Buscar caso", placeholder="señales, redes neuronales, autonomía…")
    areas_disponibles = sorted({area for caso in casos_clasificados for area in caso["areas"]})
    cats = st.multiselect("Área de inteligencia artificial", areas_disponibles, default=areas_disponibles)
    res = [
        caso for caso in casos_clasificados
        if any(area in cats for area in caso["areas"])
        and q.lower() in (caso["titulo"] + caso["desc"] + " ".join(caso["areas"])).lower()
    ]
    st.caption(f"{len(res)} caso(s)")
    for c in res:
        with st.expander(f'{c["id"]} · {c["titulo"]}'):
            st.markdown(
                f'Área principal: {pill(c["cat"], colores.get(c["cat"], "#7c3aed"))}'
                + "".join(pill(a, colores.get(a, "#7c3aed")) for a in c["areas"]),
                unsafe_allow_html=True,
            )
            st.write(c["desc"])


def pag_s04():
    st.header("Semana 04 · Ruta A* y decisión Minimax")
    t1, t2 = st.tabs(["🧭 Planificación A*", "♟️ Minimax"])
    with t1:
        n = 12
        seed = st.slider("Semilla de obstáculos", 0, 50, 7)
        dens = st.slider("Densidad de obstáculos", 0.0, 0.4, 0.22)
        rng = np.random.default_rng(seed)
        grid = (rng.random((n, n)) < dens).astype(int)
        ini, fin = (0, 0), (n - 1, n - 1)
        grid[ini], grid[fin] = 0, 0
        t0 = time.perf_counter()
        orden, ruta = astar(grid, ini, fin)
        timed("A*", t0)
        if not ruta:
            st.error("No existe ruta. Reduce la densidad de obstáculos.")
        paso = st.slider("Paso de la simulación", 1, max(len(orden), 1), len(orden))
        vis = grid.astype(float) * 4
        for p in orden[:paso]:
            vis[p] = 1
        if paso == len(orden):
            for p in ruta:
                vis[p] = 3
        vis[ini], vis[fin] = 2, 2
        fig = px.imshow(vis, color_continuous_scale=[[0, "#0a0f24"], [.25, "#7c3aed"], [.5, "#08fdd8"], [.75, "#ffb300"], [1, "#ff2e63"]], zmin=0, zmax=4)
        fig.update_layout(coloraxis_showscale=False, height=520, title="Violeta: explorado · Ámbar: ruta · Rojo: obstáculo", **PLOT)
        st.plotly_chart(fig, width="stretch")
        if ruta:
            st.success(f"Ruta óptima: {len(ruta)-1} pasos · {len(orden)} nodos explorados")
    with t2:
        st.caption("Ajusta el costo esperado (negativo = peor) de cada maniobra frente a cada escenario.")
        base = np.array([[8, 6, 7], [-9, 5, -6], [3, 4, -7]], dtype=float)
        payoff = base.copy()
        cols = st.columns(3)
        for j, col in enumerate(cols):
            for i in range(3):
                payoff[i, j] = col.slider(f"{MANIOBRAS[i]} | {RIVAL[j]}", -10, 10, int(base[i, j]), key=f"mm{i}{j}")
        mejor, peor = minimax(payoff)
        fig = px.imshow(payoff, x=RIVAL, y=MANIOBRAS, text_auto=True, color_continuous_scale="RdYlGn")
        fig.update_layout(title="Tablero de utilidades", **PLOT)
        st.plotly_chart(fig, width="stretch")
        st.success(f"Maniobra óptima (Minimax): **{MANIOBRAS[mejor]}** — peor caso garantizado: {peor[mejor]:.0f}")


def pag_s05():
    st.header("Semana 05 · Sistema híbrido (Reglas + TF-IDF)")
    q = st.text_input("🔎 Consulta de norma vial", "vehiculo debe detenerse ante pare con peatones")
    k = st.slider("Resultados", 1, 5, 3)
    if q:
        try:
            t0 = time.perf_counter()
            res = buscar_normas(q, k)
            timed("Búsqueda híbrida", t0)
            for r in res:
                with st.expander(f'Norma #{r["norma_id"]} · similitud {r["score_tfidf"]:.2f}', expanded=r is res[0]):
                    st.progress(min(r["score_tfidf"], 1.0))
                    st.markdown("".join(pill(x, "#08fdd8") for x in r["reglas_disparadas"]) or "_Sin reglas disparadas_",
                                unsafe_allow_html=True)
                    st.json(r)
        except Exception as e:
            st.error(f"Error en la búsqueda: {e}")


def responder(msg, mem):
    m = msg.lower()
    for nombre in CLASES.values():
        if nombre.lower() in m or nombre.lower().replace(" ", "") in m.replace(" ", ""):
            mem["ultima_senal"] = nombre
    if "ruta" in m:
        orden, ruta = astar(np.zeros((8, 8), int), (0, 0), (7, 7))
        return f"🛠️ `calcular_ruta()` → ruta de {len(ruta)-1} pasos, {len(orden)} nodos explorados."
    if any(w in m for w in ("regla", "acción", "accion", "qué hago", "que hago")):
        s = mem.get("ultima_senal")
        if s:
            a, p, _ = REGLAS[s]
            return f"🛠️ `consultar_regla('{s}')` → **{a}** (prioridad {p})."
        return "¿Sobre qué señal? (Pare, Ceda el Paso, Límite de Velocidad, Prohibido Girar)"
    try:
        top = buscar_normas(msg, 1)[0]
        return f"🛠️ `buscar_norma()` (similitud {top['score_tfidf']:.2f}): {top['evidencia']}"
    except Exception:
        return "No pude consultar las normas. Instala scikit-learn."


def pag_s07():
    st.header("Semana 07 · Representación numérica, simbólica y autómata")
    st.subheader("Representación numérica")
    senal = np.array([0.82, 0.05, 0.95])
    patron_pare = np.array([0.85, 0.00, 0.98])
    patron_informativa = np.array([0.05, 0.80, 0.50])
    dist_pare = float(np.linalg.norm(senal - patron_pare))
    dist_informativa = float(np.linalg.norm(senal - patron_informativa))
    c1, c2 = st.columns(2)
    with c1: kpi("Distancia a PARE", f"{dist_pare:.3f}")
    with c2: kpi("Distancia a señal informativa", f"{dist_informativa:.3f}")
    st.caption("La señal coincide con PARE cuando su distancia al patrón es menor que 0.2.")

    st.subheader("Representación simbólica e inferencia por reglas")
    hechos = {
        "color_predominante_rojo", "forma_octogonal", "texto_pare",
        "visibilidad_alta", "cerca_de_interseccion",
    }
    st.write("Hechos detectados:", ", ".join(sorted(hechos)))
    if {"color_predominante_rojo", "forma_octogonal"}.issubset(hechos):
        st.success("Regla activada: señal PARE identificada; detener el vehículo.")

    st.subheader("Autómata de secuencia de señales (FSM)")
    secuencia = st.text_input("Secuencia (P: preaviso, R: reducción, F: fin)", "PRF").upper().replace(" ", "")
    transiciones = {
        ("q0", "P"): "q1", ("q0", "R"): "q0", ("q0", "F"): "q0",
        ("q1", "R"): "q2", ("q1", "P"): "q1", ("q1", "F"): "q0",
        ("q2", "F"): "q3", ("q2", "P"): "q1", ("q2", "R"): "q2",
        ("q3", "P"): "q1", ("q3", "R"): "q0", ("q3", "F"): "q0",
    }
    if any(simbolo not in {"P", "R", "F"} for simbolo in secuencia):
        st.error("La secuencia solo puede contener P, R y F.")
    else:
        estado = "q0"
        for simbolo in secuencia:
            estado = transiciones[(estado, simbolo)]
        if estado == "q3":
            st.success(f"La secuencia `{secuencia}` completa correctamente el recorrido regulado.")
        else:
            st.info(f"La secuencia `{secuencia}` termina en el estado {estado}; no completa el recorrido.")

    st.subheader("Asistente conversacional integrado")
    ss = st.session_state
    ss.setdefault("chat", [("assistant", "Hola 👋 Soy tu agente vial. Pregunta por normas, reglas de señales o rutas.")])
    ss.setdefault("mem", {})
    for rol, txt in ss.chat:
        with st.chat_message(rol, avatar="🧑‍💻" if rol == "user" else "🤖"):
            st.markdown(txt)
    if msg := st.chat_input("Escribe tu consulta…"):
        ss.chat.append(("user", msg))
        t0 = time.perf_counter()
        try:
            resp = responder(msg, ss.mem)
        except Exception as e:
            resp = f"⚠️ Error del agente: {e}"
        ss.chat.append(("assistant", resp + f"\n\n<sub>⏱️ {(time.perf_counter()-t0)*1000:.0f} ms</sub>"))
        st.rerun()
    if st.sidebar.button("🧹 Borrar conversación"):
        ss.chat, ss.mem = [], {}
        st.rerun()


def mostrar_grafo(G):
    if Network is not None:
        net = Network(height="480px", width="100%", bgcolor="#0a0f24", font_color="#e6f1ff", directed=True)
        for n in G.nodes:
            net.add_node(n, label=n, color="#08fdd8")
        for u, v, d in G.edges(data=True):
            net.add_edge(u, v, label=d.get("rel", ""), color="#7c3aed")
        st.iframe(net.generate_html(), height=500)
    else:
        pos = nx.spring_layout(G, seed=3)
        ex, ey = [], []
        for u, v in G.edges:
            ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
        fig = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#7c3aed")),
                         go.Scatter(x=[pos[n][0] for n in G], y=[pos[n][1] for n in G], mode="markers+text",
                                    text=list(G.nodes), textposition="top center", marker=dict(size=16, color="#08fdd8"))])
        fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False, **PLOT)
        st.plotly_chart(fig, width="stretch")


def pag_s08():
    st.header("Semana 08 · Inferencia MLP, SQLite y ontología")
    if MODELO_MLP is None:
        st.warning(ERROR_MODELO_MLP)
    f = st.file_uploader(
        "Arrastra una imagen de señal de tránsito",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="s08_upload",
    )
    procesar = st.button(
        "Procesar imagen",
        key="s08_process",
        disabled=f is None or MODELO_MLP is None,
    )
    if procesar and f is not None:
        try:
            if MODELO_MLP is None:
                raise ValueError(ERROR_MODELO_MLP)
            img = Image.open(f).convert("RGB")
            img.save(UPLOADS / Path(f.name).name)
            t0 = time.perf_counter()
            pid = clasificar_imagen(img, MODELO_MLP)
            guardar_sqlite(pid)
            actualizar_ontologia(pid)
            c1, c2 = st.columns([1, 2])
            c1.image(img, caption=f.name, width="stretch")
            a, p, col = REGLAS[CLASES[pid]]
            c2.markdown(f'<div class="glass"><h3>{CLASES[pid]}</h3>{pill(p, col)}<p>{a}</p></div>', unsafe_allow_html=True)
            timed("Inferencia + SQLite + ontología", t0)
            st.toast("Registro guardado en SQLite y ontología actualizada", icon="✅")
        except Exception as e:
            st.error(f"No se pudo procesar la imagen: {e}")
    st.subheader("Últimos registros (SQLite)")
    st.dataframe(leer_sqlite(), width="stretch", hide_index=True)
    st.subheader("Grafo ontológico")
    mostrar_grafo(cargar_ontologia())


def pag_s09():
    st.header("Semana 09 · Visión artificial: Canny, Otsu y regiones")
    if canny is None:
        st.error("Instala scikit-image: `pip install scikit-image`")
        return
    f = st.file_uploader("Imagen a analizar", type=["png", "jpg", "jpeg", "bmp", "webp"], key="v09")
    sigma = st.slider("Sigma (Canny)", 0.5, 6.0, 2.0, 0.1)
    if not f:
        st.info("Sube una imagen para comenzar.")
        return
    try:
        img = Image.open(f)
        t0 = time.perf_counter()
        gris, bordes, mascara, color, umbral, n = vision(img, sigma)
        timed("Pipeline de visión", t0)
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Umbral de Otsu", f"{umbral:.3f}")
        with c2: kpi("Regiones conectadas", n)
        with c3: kpi("Sigma", f"{sigma:.1f}")
        vistas = {"Grises": gris, "Canny": bordes.astype(float), "Otsu": mascara.astype(float), "Regiones": color}
        orig = np.array(img.convert("RGB").resize((256, 256)))
        tabs = st.tabs(list(vistas))
        for tab, (nombre, v) in zip(tabs, vistas.items()):
            with tab:
                a, b = st.columns(2)
                a.image(orig, caption="Original", width="stretch")
                b.image(v, caption=nombre, width="stretch", clamp=True)
    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")


def pag_s10():
    st.header("Semana 10 · Reconocimiento de imágenes y descriptores LBP")
    st.caption("Extrae características de regiones, intensidad y textura para una señal.")
    if local_binary_pattern is None:
        st.error("Instala scikit-image: `pip install scikit-image`")
        return

    f = st.file_uploader("Imagen de señal a analizar", type=["png", "jpg", "jpeg", "bmp", "webp"], key="v10")
    if not f:
        st.info("Sube una imagen para extraer sus características.")
        return

    try:
        img = Image.open(f).convert("L")
        gris = np.array(img, dtype=np.uint8)
        t0 = time.perf_counter()
        from semana10_reconocimiento import extraer_vector_caracteristicas_senal

        vector, mascara = extraer_vector_caracteristicas_senal(gris)
        ruta_vector = ARTIFACTS / "semana10_features.npy"
        np.save(ruta_vector, vector)
        timed("Extracción de características", t0)

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Área media", f"{vector[0]:.1f} px")
        with c2: kpi("Desviación de área", f"{vector[1]:.1f} px")
        with c3: kpi("Regiones", f"{int(vector[2])}")
        with c4: kpi("Características", len(vector))

        izquierda, derecha = st.columns(2)
        izquierda.image(gris, caption="Imagen en escala de grises", width="stretch")
        derecha.image(mascara, caption="Máscara segmentada con Otsu", width="stretch")

        datos = pd.DataFrame({
            "Índice": list(range(3, len(vector))),
            "Valor": vector[3:],
            "Descriptor": ["Intensidad"] * 32 + ["Textura LBP"] * len(vector[35:]),
        })
        fig = px.bar(datos, x="Índice", y="Valor", facet_col="Descriptor",
                     title="Histogramas de intensidad y textura LBP")
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, width="stretch")
        st.success(f"Vector de características guardado en `{ruta_vector.relative_to(ROOT)}`.")
    except Exception as e:
        st.error(f"No se pudieron extraer las características: {e}")


def cargar_datos_benchmark():
    clases = {
        0: "0_pare",
        1: "1_ceda_el_paso",
        2: "2_limite_velocidad",
        3: "3_prohibido_girar",
    }
    imagenes, etiquetas = [], []
    base = ROOT / "data" / "img"
    for clase, carpeta in clases.items():
        ruta_clase = base / carpeta
        if not ruta_clase.is_dir():
            continue
        for ruta_imagen in ruta_clase.iterdir():
            if ruta_imagen.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            with Image.open(ruta_imagen) as imagen:
                vector = np.asarray(imagen.convert("L").resize((32, 32)), dtype=np.float32).ravel() / 255.0
            imagenes.append(vector)
            etiquetas.append(clase)

    conteos = np.bincount(etiquetas, minlength=4) if etiquetas else np.zeros(4, dtype=int)
    usa_sinteticos = len(imagenes) < 20 or np.any(conteos < 2)
    if usa_sinteticos:
        rng = np.random.default_rng(42)
        return rng.random((120, 1024)), rng.integers(0, 4, size=120), True
    return np.asarray(imagenes), np.asarray(etiquetas), False


def pag_s11():
    st.header("Semana 11 · Benchmark de Tres Modelos y Decisión Técnica (ADR)")
    st.caption("Compara tres modelos sobre el mismo conjunto de imágenes y partición de entrenamiento.")
    if any(dependency is None for dependency in (
        train_test_split, make_pipeline, StandardScaler, LogisticRegression,
        RandomForestClassifier, MLPClassifier, accuracy_score,
    )):
        st.error("Instala scikit-learn para ejecutar el benchmark.")
        return

    if st.button("🚀 Ejecutar Benchmark de Modelos"):
        with st.spinner("Entrenando y evaluando modelos bajo el mismo protocolo..."):
            try:
                X, y, usa_sinteticos = cargar_datos_benchmark()
                if usa_sinteticos:
                    st.warning(
                        "No hay al menos 20 imágenes válidas con muestras de las cuatro clases en "
                        "`data/img`; se usarán datos sintéticos para completar la demostración."
                    )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )

                models = {
                    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=42)),
                    "forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                    "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42))
                }

                rows = []
                for name, model in models.items():
                    start = perf_counter()
                    model.fit(X_train, y_train)
                    elapsed = perf_counter() - start
                    acc = accuracy_score(y_test, model.predict(X_test))
                    rows.append({
                        "modelo": name,
                        "accuracy": round(acc, 4),
                        "fit_seconds": round(elapsed, 4)
                    })

                df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
                df.to_csv(ARTIFACTS / "benchmark.csv", index=False)

                adr_content = f"""# ADR Semana 11 - Selección de Modelo

## Contexto
Se evaluaron tres modelos de clasificación para el detector de señales de tránsito bajo un protocolo estricto (mismo dataset, partición y semilla). El dataset local se carga desde `data/img`; si no contiene suficientes imágenes de las cuatro clases, se utiliza un conjunto sintético de respaldo.

## Evidencia
Resultados extraídos de `artifacts/benchmark.csv`:
- **Mejor Modelo**: {df.iloc[0]['modelo']} con Accuracy {df.iloc[0]['accuracy']} y tiempo {df.iloc[0]['fit_seconds']}s.

## Decisión
Se selecciona **{df.iloc[0]['modelo']}** por ofrecer el balance idóneo entre precisión predictiva y eficiencia temporal.
"""
                with open(REPORTS / "semana11_adr.md", "w", encoding="utf-8") as f:
                    f.write(adr_content)

                st.success("¡Benchmark ejecutado con éxito! Artefactos guardados.")
                st.dataframe(df, width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"Error al ejecutar el benchmark: {e}")

    csv_path = ARTIFACTS / "benchmark.csv"
    if csv_path.exists():
        st.subheader("📊 Último Benchmark Registrado (`benchmark.csv`)")
        df_saved = pd.read_csv(csv_path)
        st.dataframe(df_saved, width="stretch", hide_index=True)

    adr_path = REPORTS / "semana11_adr.md"
    if adr_path.exists():
        with st.expander("📄 Ver Architecture Decision Record (ADR)"):
            st.markdown(adr_path.read_text(encoding="utf-8"))


def pag_s13():
    st.header("Semana 13 · Motor de Inferencia Lógica y Reglas de Tránsito")
    st.caption("Implementación de ingeniería del conocimiento: hechos, reglas tipo Horn y encadenamiento hacia adelante[cite: 1].")
    
    if reglas_senales is None:
        st.error("No se pudo cargar el módulo `semana13_motor_inferencia_transito.py` en `src/`.")
        return

    st.subheader("📋 Catálogo de Reglas del Sistema (Reglas de Horn)")
    for regla in reglas_senales:
        premisas_str = " ∧ ".join(list(regla.premisas))
        st.markdown(f"- **{regla.nombre}**: `{premisas_str}` $\\rightarrow$ **{regla.conclusion}**")

    st.markdown("### 🧪 Configuración de Evidencias / Hechos Iniciales")
    
    hechos_disponibles = [
        "senal_pare_detectada",
        "linea_detencion_visible",
        "vehiculo_en_movimiento",
        "via_despejada_tras_3s"
    ]
    
    hechos_seleccionados = st.multiselect(
        "Selecciona los hechos detectados por el sistema de control:",
        options=hechos_disponibles,
        default=["senal_pare_detectada", "linea_detencion_visible", "vehiculo_en_movimiento"]
    )

    if st.button("🚀 Ejecutar Motor de Inferencia hacia Adelante"):
        t0 = time.perf_counter()
        finales, trazas = inferir_transito(hechos_seleccionados, reglas_senales)
        timed("Inferencia lógica", t0)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Hechos Iniciales")
            st.write(list(hechos_seleccionados))
        with c2:
            st.markdown("#### Hechos Derivados Finales")
            st.write(list(finales))
            
        st.markdown("#### 🔍 Trazas de Reglas Activadas (Explicabilidad)")
        if trazas:
            st.success(f"Reglas ejecutadas secuencialmente: `{' -> '.join(trazas)}`")
        else:
            st.warning("No se activó ninguna regla con los hechos iniciales proporcionados.")
            
        if "accion_frenar_emergencia" in finales:
            st.toast("¡Acción de frenado de emergencia demostrada con éxito!", icon="🚨")


def pag_s14():
    st.header("Semana 14 · Red Semántica de Señales de Tránsito")
    st.caption(
        "Explora relaciones semánticas, jerarquías de señales y propiedades heredadas."
    )

    hechos = [
        ("pare", "es_un", "senal_reglamentaria"),
        ("ceda_el_paso", "es_un", "senal_reglamentaria"),
        ("senal_reglamentaria", "es_un", "senal_transito"),
        ("limite_velocidad", "es_un", "senal_reglamentaria"),
        ("senal_reglamentaria", "es_un", "senal_transito"),
        ("senal_reglamentaria", "color_borde", "rojo"),
        ("senal_reglamentaria", "requiere_accion", "detencion_o_ceder"),
        ("pare", "requiere_accion", "detencion_total_inmediata"),
        ("limite_velocidad", "color_borde", "rojo_con_negro"),
        ("limite_velocidad", "requiere_accion", "ajustar_velocidad"),
    ]
    red = nx.DiGraph()
    red.add_edges_from(
        (sujeto, objeto, {"relacion": relacion})
        for sujeto, relacion, objeto in hechos
    )

    senales = ["pare", "ceda_el_paso", "limite_velocidad"]
    seleccion = st.selectbox(
        "Señal para consultar",
        senales,
        format_func=lambda valor: valor.replace("_", " ").capitalize(),
    )

    relaciones_taxonomicas = nx.DiGraph(
        (sujeto, objeto)
        for sujeto, relacion, objeto in hechos
        if relacion == "es_un"
    )
    tipos = [
        objeto
        for objeto in relaciones_taxonomicas
        if objeto != seleccion and nx.has_path(relaciones_taxonomicas, seleccion, objeto)
    ]
    propiedades = []
    sujetos_conocidos = [seleccion, *tipos]
    for sujeto, relacion, objeto in hechos:
        if sujeto in sujetos_conocidos and relacion != "es_un":
            propiedades.append({
                "Propiedad": relacion.replace("_", " ").capitalize(),
                "Valor": objeto.replace("_", " "),
                "Origen": "Directa" if sujeto == seleccion else f"Heredada de {sujeto.replace('_', ' ')}",
            })

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Jerarquía")
        if tipos:
            st.success(
                f"**{seleccion.replace('_', ' ')}** es una señal de tránsito "
                f"({ ' → '.join(tipo.replace('_', ' ') for tipo in tipos) })."
            )
        else:
            st.warning("No se encontró una jerarquía para esta señal.")
    with c2:
        st.markdown("#### Propiedades propias y heredadas")
        if propiedades:
            st.dataframe(pd.DataFrame(propiedades), width="stretch", hide_index=True)
        else:
            st.info("No hay propiedades definidas para esta señal.")

    st.subheader("🕸️ Grafo de relaciones")
    posiciones = nx.spring_layout(red, seed=14)
    aristas_x, aristas_y = [], []
    anotaciones = []
    for sujeto, objeto, datos in red.edges(data=True):
        x0, y0 = posiciones[sujeto]
        x1, y1 = posiciones[objeto]
        aristas_x.extend([x0, x1, None])
        aristas_y.extend([y0, y1, None])
        anotaciones.append({
            "x": x1,
            "y": y1,
            "ax": x0,
            "ay": y0,
            "xref": "x",
            "yref": "y",
            "axref": "x",
            "ayref": "y",
            "showarrow": True,
            "arrowhead": 3,
            "arrowsize": 1,
            "arrowwidth": 1.2,
            "arrowcolor": "#08fdd8",
            "text": datos["relacion"].replace("_", " "),
            "font": {"color": "#e6f1ff", "size": 10},
            "bgcolor": "#0a0f24",
        })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aristas_x,
        y=aristas_y,
        mode="lines",
        line={"color": "#08fdd855", "width": 1},
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[posiciones[nodo][0] for nodo in red.nodes],
        y=[posiciones[nodo][1] for nodo in red.nodes],
        mode="markers+text",
        text=[nodo.replace("_", " ") for nodo in red.nodes],
        textposition="bottom center",
        textfont={"color": "#e6f1ff", "size": 11},
        marker={
            "size": 20,
            "color": [
                "#ff2e63" if nodo == seleccion else
                "#7c3aed" if any(nodo == tipo for tipo in tipos) else
                "#08fdd8"
                for nodo in red.nodes
            ],
            "line": {"width": 1, "color": "#e6f1ff"},
        },
        hovertext=[nodo.replace("_", " ") for nodo in red.nodes],
        hoverinfo="text",
    ))
    fig.update_layout(
        height=560,
        showlegend=False,
        annotations=anotaciones,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 20, "b": 30},
        **PLOT,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Cada flecha muestra una relación; `es un` representa la jerarquía taxonómica.")


def pag_s15():
    st.header("Semana 15 · Base de Conocimiento Vial")
    st.caption("Consulta artículos normativos validados y administra la aprobación de borradores.")

    if (
        buscar_articulos_validos is None
        or cargar_datos_iniciales is None
        or conectar is None
        or inicializar_base_conocimiento is None
        or validar_articulo_por_id is None
    ):
        st.error("No se pudo cargar `semana15_conocimiento_vial.py` desde `src/`.")
        return

    ruta_db = str(ROOT / "data" / "senales_conocimiento.db")
    if st.button("🗄️ Inicializar y cargar datos de ejemplo"):
        try:
            inicializar_base_conocimiento(ruta_db)
            cargar_datos_iniciales(ruta_db)
        except sqlite3.Error as error:
            st.error(f"No se pudo preparar la base de conocimiento: {error}")
            return
        st.success("Base de conocimiento preparada.")

    try:
        with closing(conectar(ruta_db)) as conexion:
            dominios = [
                fila[0]
                for fila in conexion.execute(
                    "SELECT nombre FROM dominio_vial ORDER BY nombre"
                )
            ]
            borradores = conexion.execute(
                """
                SELECT id, codigo_senal, titulo
                FROM articulo_normativo
                WHERE estado = 'BORRADOR'
                ORDER BY id
                """
            ).fetchall()
    except sqlite3.OperationalError:
        st.info("Inicializa la base de conocimiento para crear sus tablas y datos de ejemplo.")
        return
    except sqlite3.Error as error:
        st.error(f"No se pudo consultar la base de conocimiento: {error}")
        return

    if not dominios:
        st.warning("No hay dominios viales registrados.")
        return

    dominio = st.selectbox("Dominio vial", dominios)
    try:
        articulos = buscar_articulos_validos(dominio, ruta_db)
    except sqlite3.Error as error:
        st.error(f"No se pudieron consultar los artículos validados: {error}")
        return

    st.subheader("Artículos validados")
    if articulos:
        st.dataframe(
            pd.DataFrame(
                articulos,
                columns=["Señal", "Título", "Contenido", "Responsable", "Fuente", "Estado"],
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No hay artículos validados para este dominio.")

    st.subheader("Aprobación de borradores")
    if not borradores:
        st.info("No hay borradores pendientes de aprobación.")
        return

    articulo_id = st.selectbox(
        "Borrador pendiente",
        options=[fila[0] for fila in borradores],
        format_func=lambda id_articulo: next(
            f"{senal} · {titulo}"
            for id_fila, senal, titulo in borradores
            if id_fila == id_articulo
        ),
    )
    if st.button("✅ Validar artículo seleccionado"):
        try:
            validar_articulo_por_id(articulo_id, ruta_db)
        except sqlite3.Error as error:
            st.error(f"No se pudo validar el artículo: {error}")
        else:
            st.success("Artículo validado.")
            st.rerun()


def pag_s16():
    st.header("Semana 16 · Detector y Trazabilidad de Referencias")
    st.caption("Consulta la guía de tránsito y recibe respuestas solo de referencias aprobadas.")

    if REFERENCIAS_VIALES is None or motor_de_referencia is None:
        st.error("No se pudo cargar `semana16_detector_transito_pruebas.py` desde `src/`.")
        return

    with st.form("consulta_referencia_s16"):
        consulta = st.text_input(
            "Descripción de la señal o situación vial",
            placeholder="Ejemplo: señal de pare octagonal",
        )
        consultar = st.form_submit_button("🔎 Consultar referencias")

    if consultar:
        if not consulta.strip():
            st.warning("Escribe una descripción para realizar la consulta.")
        else:
            resultado = motor_de_referencia(consulta)
            if resultado["estado"] == "ENCONTRADA":
                st.success(f"Referencia encontrada: {resultado['titulo']}")
                st.write(resultado["respuesta"])
                st.caption(f"Fuente: {resultado['id']} · {resultado['fuente']}")
            else:
                st.warning(resultado["respuesta"])
                st.caption(f"Estado: {resultado['estado']} · {resultado['fuente']}")

    referencias_aprobadas = [
        {
            "ID": referencia["id"],
            "Título": referencia["titulo"],
            "Fuente": referencia["fuente"],
        }
        for referencia in REFERENCIAS_VIALES
        if referencia["validada"]
    ]
    st.subheader("Referencias aprobadas")
    st.dataframe(
        pd.DataFrame(referencias_aprobadas),
        width="stretch",
        hide_index=True,
    )
    st.caption("Las referencias no aprobadas se excluyen del motor de consulta.")


def pag_integrado():
    st.title("🚦 Sistema Inteligente de Control Vehicular")
    st.caption("Clasificación, reglas de tránsito, consulta normativa y registro de eventos.")
    if MODELO_MLP is None:
        st.warning(
            f"No se realizarán predicciones automáticas. {ERROR_MODELO_MLP} "
            "El modelo compatible se genera desde `src/semana08_red_ontologia.py`."
        )

    f = st.file_uploader("⚡ Pipeline completo: sube una señal", type=["png", "jpg", "jpeg", "webp"], key="full")
    procesar = st.button(
        "Procesar imagen",
        key="full_process",
        disabled=f is None or MODELO_MLP is None,
    )
    if procesar and f is not None and MODELO_MLP is not None:
        try:
            imagen_bytes = f.getvalue()
            img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
            t0 = time.perf_counter()
            pid = clasificar_imagen(img, MODELO_MLP)
            norma = None
            error_norma = None
            try:
                resultados_norma = buscar_normas(CLASES[pid], 1)
                norma = resultados_norma[0] if resultados_norma else None
            except RuntimeError as error:
                error_norma = str(error)

            referencia = (
                motor_de_referencia(CLASES[pid])
                if motor_de_referencia is not None
                else None
            )
            articulos = []
            error_base = None
            dominio_por_clase = {
                0: "Regulación Prioritaria",
                1: "Regulación Prioritaria",
                2: "Control de Velocidad",
            }
            dominio = dominio_por_clase.get(pid)
            ruta_base = ROOT / "data" / "senales_conocimiento.db"
            if dominio and buscar_articulos_validos is not None and ruta_base.is_file():
                try:
                    articulos = buscar_articulos_validos(dominio, str(ruta_base))
                except sqlite3.Error as error:
                    error_base = f"No se pudo consultar la base de conocimiento: {error}"
            elif dominio:
                error_base = "Base de conocimiento no inicializada; prepárala en la página de Semana 15."

            datos_vision = None
            error_vision = None
            if canny is not None:
                try:
                    *_, umbral, regiones = vision(img, 2.0)
                    datos_vision = {"umbral": float(umbral), "regiones": int(regiones)}
                except (ValueError, RuntimeError) as error:
                    error_vision = f"No se pudo completar el análisis de visión: {error}"

            actualizar_ontologia(pid)
            guardar_sqlite(pid)
            st.session_state["integrated_result"] = {
                "imagen": imagen_bytes,
                "nombre_archivo": f.name,
                "clase_id": pid,
                "norma": norma,
                "error_norma": error_norma,
                "referencia": referencia,
                "articulos": articulos,
                "error_base": error_base,
                "vision": datos_vision,
                "error_vision": error_vision,
                "tiempo_ms": (time.perf_counter() - t0) * 1000,
            }
        except (OSError, ValueError, RuntimeError, sqlite3.Error, nx.NetworkXException) as error:
            st.error(f"No se pudo completar el pipeline integrado: {error}")

    registros = leer_sqlite(1000)
    c = st.columns(4)
    with c[0]: kpi("Señales procesadas", len(registros))
    with c[1]: kpi("Normas indexadas", len(NORMAS))
    with c[2]: kpi("Casos viales", len(CASOS))
    with c[3]: kpi("Clases detectables", len(CLASES))

    resultado = st.session_state.get("integrated_result")
    if resultado and f is not None and resultado["imagen"] == f.getvalue():
        img = Image.open(io.BytesIO(resultado["imagen"]))
        pid = resultado["clase_id"]
        accion, prioridad, color = REGLAS[CLASES[pid]]
        izquierda, derecha = st.columns([1, 2])
        izquierda.image(img, caption=resultado["nombre_archivo"], width="stretch")
        derecha.markdown(
            f'<div class="glass"><h2>{CLASES[pid]}</h2>{pill(prioridad, color)}'
            f'<h4>{accion}</h4></div>',
            unsafe_allow_html=True,
        )
        derecha.caption(f"Registro SQLite: actualizado · Ontología: actualizada")
        st.caption(f"⏱️ Pipeline integrado: {resultado['tiempo_ms']:.1f} ms")

        st.subheader("Evidencia normativa")
        if resultado["referencia"] is not None:
            referencia = resultado["referencia"]
            if referencia["estado"] == "ENCONTRADA":
                st.success(
                    f'{referencia["id"]} · {referencia["titulo"]} · {referencia["fuente"]}'
                )
                st.write(referencia["respuesta"])
            else:
                st.warning(referencia["respuesta"])
        else:
            st.info("El motor de referencias de la Semana 16 no está disponible.")

        if pid == 0:
            st.subheader("Inferencia lógica · Semana 13")
            if inferir_transito is None or reglas_senales is None:
                st.info("El motor lógico de la Semana 13 no está disponible.")
            else:
                evidencias = st.multiselect(
                    "Evidencias adicionales observadas",
                    options=[
                        "linea_detencion_visible",
                        "vehiculo_en_movimiento",
                        "via_despejada_tras_3s",
                    ],
                    format_func=lambda hecho: hecho.replace("_", " ").capitalize(),
                    key="integrated_facts",
                )
                hechos_iniciales = {"senal_pare_detectada", *evidencias}
                hechos_finales, trazas = inferir_transito(
                    hechos_iniciales,
                    reglas_senales,
                )
                st.write("Hechos derivados:", sorted(hechos_finales))
                if trazas:
                    st.caption(f"Reglas activadas: {' → '.join(trazas)}")
                else:
                    st.info("Con las evidencias seleccionadas no se activa una regla adicional.")

        if resultado["articulos"]:
            st.dataframe(
                pd.DataFrame(
                    resultado["articulos"],
                    columns=["Señal", "Título", "Contenido", "Responsable", "Fuente", "Estado"],
                ),
                width="stretch",
                hide_index=True,
            )
        elif resultado["error_base"]:
            st.info(resultado["error_base"])
        else:
            st.info("No hay artículos validados para el dominio de esta señal.")

        if resultado["norma"] is not None:
            st.json(resultado["norma"])
        elif resultado["error_norma"]:
            st.warning(resultado["error_norma"])
        else:
            st.info("No se encontró una norma textual relacionada.")

        if resultado["vision"] is not None:
            st.caption(
                f"Visión artificial: Otsu={resultado['vision']['umbral']:.3f} · "
                f"{resultado['vision']['regiones']} regiones"
            )
        elif resultado["error_vision"]:
            st.warning(resultado["error_vision"])

    if len(registros):
        fig = px.pie(registros, names="etiqueta_nombre", hole=.55, title="Distribución de señales detectadas",
                     color_discrete_sequence=["#08fdd8", "#7c3aed", "#ffb300", "#ff2e63"])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, width="stretch")


ROUTER = {
    MODULOS[0]: pag_integrado,
    MODULOS[1]: pag_s02,
    MODULOS[2]: pag_s03,
    MODULOS[3]: pag_s04,
    MODULOS[4]: pag_s05,
    MODULOS[5]: pag_s07,
    MODULOS[6]: pag_s08,
    MODULOS[7]: pag_s09,
    MODULOS[8]: pag_s10,
    MODULOS[9]: pag_s11,
    MODULOS[10]: pag_s13,
    MODULOS[11]: pag_s14,
    MODULOS[12]: pag_s15,
    MODULOS[13]: pag_s16
}

try:
    ROUTER[pagina]()
except Exception as e:
    st.error(f"Error inesperado en el módulo: {e}")