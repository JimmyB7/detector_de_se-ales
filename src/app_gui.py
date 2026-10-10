"""
Sistema Inteligente de Control Vehicular y Detector de Señales de Tránsito
Dashboard Streamlit (Dark Glassmorphism / Cyberpunk) - Semanas 02 a 10

Instalar:  pip install streamlit plotly pandas numpy pillow networkx scikit-learn scikit-image pyvis
Ejecutar:  streamlit run dashboard_app.py     (desde ia_semestre/ o ia_semestre/src/)
"""
import heapq, json, pickle, re, sqlite3, sys, time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
try:
    from skimage.feature import canny, local_binary_pattern
    from skimage.filters import threshold_otsu
    from skimage.measure import label as sk_label, regionprops
    from skimage.color import label2rgb
except ImportError:
    canny = None
    local_binary_pattern = None
try:
    from pyvis.network import Network
except ImportError:
    Network = None

# ---------------------------------------------------------------- Rutas
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "src" else HERE
ARTIFACTS, UPLOADS = ROOT / "artifacts", ROOT / "uploads"
for p in (ARTIFACTS, UPLOADS):
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
    try:
        with open(ruta, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def clasificar_imagen(img: Image.Image, mlp):
    """Regla de color rojo/centro blanco (PARE vs CEDA) + fallback MLP."""
    arr = np.array(img.convert("RGB").resize((64, 64)), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    rojo = ((r > 120) & (g < 90) & (b < 90)).mean()
    c = arr[16:48, 16:48]
    blanco = ((c[..., 0] > 180) & (c[..., 1] > 180) & (c[..., 2] > 180)).mean()
    if rojo > 0.15:
        return 1 if blanco > 0.35 else 0
    if mlp is None:
        return 2
    g32 = np.array(img.convert("L").resize((32, 32)), dtype=np.float32) / 255.0
    if g32.mean() > 0.5:
        g32 = 1.0 - g32
    return int(mlp.predict(g32.reshape(1, -1))[0])


def guardar_sqlite(pid):
    with sqlite3.connect(ARTIFACTS / "imagenes.db") as con:
        con.execute("CREATE TABLE IF NOT EXISTS imagenes_senales(id INTEGER PRIMARY KEY, "
                    "etiqueta_id INTEGER, etiqueta_nombre TEXT, split TEXT)")
        con.execute("INSERT INTO imagenes_senales (etiqueta_id,etiqueta_nombre,split) VALUES (?,?,?)",
                    (pid, CLASES[pid], "subida_usuario"))


def leer_sqlite(n=15):
    try:
        with sqlite3.connect(ARTIFACTS / "imagenes.db") as con:
            return pd.read_sql_query(f"SELECT * FROM imagenes_senales ORDER BY id DESC LIMIT {n}", con)
    except Exception:
        return pd.DataFrame(columns=["id", "etiqueta_id", "etiqueta_nombre", "split"])


def actualizar_ontologia(pid):
    ruta = ARTIFACTS / "ontologia.graphml"
    try:
        G = nx.read_graphml(ruta) if ruta.exists() else nx.DiGraph()
    except Exception:
        G = nx.DiGraph()
    G.add_edge("imagen_usuario", "prediccion", rel="genera")
    G.add_edge("prediccion", ONTO[pid], rel="asigna_clase")
    nx.write_graphml(G, ruta)
    return G


def cargar_ontologia():
    ruta = ARTIFACTS / "ontologia.graphml"
    if ruta.exists():
        try:
            return nx.read_graphml(ruta)
        except Exception:
            pass
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


def extraer_caracteristicas_s10(img: Image.Image, min_area: int = 50, radius: int = 2):
    """Procesamiento Semana 10: Segmentación Otsu + Regiones + Histograma Intensidad + LBP"""
    gris_int = np.array(img.convert("L").resize((256, 256)), dtype=np.uint8)
    gris_norm = gris_int.astype(np.float32) / 255.0
    
    # 1. Segmentación y Regiones
    umbral = threshold_otsu(gris_norm)
    mascara = gris_norm > umbral
    etiquetas = sk_label(mascara)
    regiones = [r for r in regionprops(etiquetas) if r.area > min_area]
    
    areas = np.array([r.area for r in regiones], dtype=float)
    area_mean = float(areas.mean()) if len(areas) else 0.0
    area_std = float(areas.std()) if len(areas) else 0.0
    
    # 2. Histograma Intensidad Global (32 Bins)
    intensity_hist, _ = np.histogram(gris_int.ravel(), bins=32, range=(0, 256), density=True)
    
    # 3. Textura LBP
    points = 8 * radius
    lbp = local_binary_pattern(gris_int, points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), density=True)
    
    # Vector Concatenado
    vector_features = np.concatenate([
        [area_mean, area_std, len(regiones)],
        intensity_hist,
        lbp_hist
    ])
    
    # Guardar Artefactos
    np.save(ARTIFACTS / "semana10_features.npy", vector_features)
    
    return gris_int, mascara, regiones, area_mean, area_std, intensity_hist, lbp, lbp_hist, vector_features


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


# ---------------------------------------------------------------- Datos demo
CASOS = [
    {"id": "C-001", "titulo": "Vehículo ignora señal de pare en intersección", "cat": "Infracción", "areas": ["Intersección", "Señalización"], "desc": "Un vehículo no se detiene ante PARE y casi colisiona con un peatón."},
    {"id": "C-002", "titulo": "Exceso de velocidad en zona escolar", "cat": "Velocidad", "areas": ["Zona escolar", "Peatones"], "desc": "Se supera el límite de 30 km/h en horario de entrada escolar."},
    {"id": "C-003", "titulo": "Giro prohibido en avenida principal", "cat": "Infracción", "areas": ["Avenida", "Señalización"], "desc": "Giro a la izquierda donde la señal lo prohíbe."},
    {"id": "C-004", "titulo": "No cede el paso en rotonda", "cat": "Prioridad", "areas": ["Rotonda", "Intersección"], "desc": "Se ingresa a la rotonda sin ceder al tráfico circulante."},
]
CAT_COLOR = {"Infracción": "#ff2e63", "Velocidad": "#ffb300", "Prioridad": "#08fdd8"}
NORMAS = [
    "Todo conductor debe detenerse completamente ante una señal de PARE y ceder el paso a peatones.",
    "Ceda el paso: reduzca la velocidad y permita la circulación de vehículos con prioridad.",
    "El límite de velocidad en zona escolar es de 30 km/h; respete la señalización vertical.",
    "Está prohibido girar a la izquierda o realizar retorno donde exista señal de prohibido girar.",
    "En una rotonda tiene prioridad el vehículo que ya circula dentro de ella.",
    "El conductor debe ceder el paso al peatón en cruces señalizados y zonas peatonales.",
]
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
st.sidebar.caption("Dashboard del semestre · Semanas 02-10")
MODULOS = ["🏠 Sistema Integrado", "S02 · Clasificación y Reglas", "S03 · Taxonomía de Casos",
           "S04 · A* y Minimax", "S05 · Híbrido TF-IDF", "S07 · Agente Conversacional",
           "S08 · MLP, SQLite y Ontología", "S09 · Visión Artificial", "S10 · Reconocimiento e Histogramas"]
pagina = st.sidebar.radio("Navegación", MODULOS)
st.sidebar.markdown("---")
st.sidebar.caption(f"Modelo MLP: {'✅ cargado' if cargar_modelo() else '⚠️ no encontrado (solo reglas)'}")
st.sidebar.caption(f"scikit-image: {'✅' if canny else '❌'} · pyvis: {'✅' if Network else '❌ (usa Plotly)'}")

# ================================================================ PÁGINAS
def pag_s02():
    st.header("Semana 02 · Clasificación visual y motor de reglas")
    cm = np.array([[48, 2, 0, 0], [3, 45, 1, 1], [0, 1, 47, 2], [1, 0, 2, 47]])
    tp = np.diag(cm)
    acc = tp.sum() / cm.sum()
    prec = np.mean(tp / cm.sum(axis=0))
    rec = np.mean(tp / cm.sum(axis=1))
    c1, c2, c3 = st.columns(3)
    with c1: kpi("Accuracy", f"{acc:.1%}")
    with c2: kpi("Precisión (macro)", f"{prec:.1%}")
    with c3: kpi("Recuerdo (macro)", f"{rec:.1%}")
    nombres = list(CLASES.values())
    fig = px.imshow(cm, x=nombres, y=nombres, text_auto=True, color_continuous_scale="Tealgrn",
                    labels=dict(x="Predicho", y="Real"))
    fig.update_layout(title="Matriz de confusión (datos de ejemplo)", **PLOT)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Simulación de fotogramas + motor de reglas")
    frames = st.multiselect("Secuencia de fotogramas detectados", nombres, default=["Limite Velocidad", "Ceda el Paso", "Pare"])
    cols = st.columns(max(len(frames), 1))
    for i, (col, s) in enumerate(zip(cols, frames)):
        accion, prio, color = REGLAS[s]
        col.markdown(f'<div class="glass"><small>Fotograma {i+1}</small><h4>{s}</h4>{pill(prio, color)}'
                     f'<p>{accion}</p></div>', unsafe_allow_html=True)
    if any(REGLAS[s][1] == "Crítico" for s in frames):
        st.toast("Prioridad CRÍTICA detectada: frenado inmediato", icon="🚨")


def pag_s03():
    st.header("Semana 03 · Análisis taxonómico de casos viales")
    q = st.text_input("🔎 Buscar caso", placeholder="pare, escolar, rotonda…")
    cats = st.multiselect("Categoría", list(CAT_COLOR), default=list(CAT_COLOR))
    res = [c for c in CASOS if c["cat"] in cats and q.lower() in (c["titulo"] + c["desc"] + " ".join(c["areas"])).lower()]
    st.caption(f"{len(res)} caso(s)")
    for c in res:
        with st.expander(f'{c["id"]} · {c["titulo"]}'):
            st.markdown(pill(c["cat"], CAT_COLOR[c["cat"]]) + "".join(pill(a, "#7c3aed") for a in c["areas"]),
                        unsafe_allow_html=True)
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
        st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(fig, use_container_width=True)
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
    st.header("Semana 07 · Agente conversacional adaptativo")
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
        components.html(net.generate_html(), height=500)
    else:
        pos = nx.spring_layout(G, seed=3)
        ex, ey = [], []
        for u, v in G.edges:
            ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
        fig = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#7c3aed")),
                         go.Scatter(x=[pos[n][0] for n in G], y=[pos[n][1] for n in G], mode="markers+text",
                                    text=list(G.nodes), textposition="top center", marker=dict(size=16, color="#08fdd8"))])
        fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False, **PLOT)
        st.plotly_chart(fig, use_container_width=True)


def pag_s08():
    st.header("Semana 08 · Inferencia MLP, SQLite y ontología")
    f = st.file_uploader("Arrastra una imagen de señal de tránsito", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if f:
        try:
            img = Image.open(f)
            img.convert("RGB").save(UPLOADS / Path(f.name).name)
            t0 = time.perf_counter()
            pid = clasificar_imagen(img, cargar_modelo())
            guardar_sqlite(pid)
            actualizar_ontologia(pid)
            c1, c2 = st.columns([1, 2])
            c1.image(img, caption=f.name, use_container_width=True)
            a, p, col = REGLAS[CLASES[pid]]
            c2.markdown(f'<div class="glass"><h3>{CLASES[pid]}</h3>{pill(p, col)}<p>{a}</p></div>', unsafe_allow_html=True)
            timed("Inferencia + SQLite + ontología", t0)
            st.toast("Registro guardado en SQLite y ontología actualizada", icon="✅")
        except Exception as e:
            st.error(f"No se pudo procesar la imagen: {e}")
    st.subheader("Últimos registros (SQLite)")
    st.dataframe(leer_sqlite(), use_container_width=True, hide_index=True)
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
                a.image(orig, caption="Original", use_container_width=True)
                b.image(v, caption=nombre, use_container_width=True, clamp=True)
    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")


def pag_s10():
    st.header("Semana 10 · Reconocimiento: Histogramas, Regiones y Texturas LBP")
    if local_binary_pattern is None:
        st.error("Instala scikit-image: `pip install scikit-image`")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        f = st.file_uploader("Subir imagen de señal", type=["png", "jpg", "jpeg", "bmp", "webp"], key="v10")
        min_area = st.number_input("Filtro área mínima (px)", min_value=10, max_value=500, value=50, step=10)
        radius = st.slider("Radio LBP", min_value=1, max_value=5, value=2)

    if not f:
        st.info("Sube una imagen para extraer su vector de características de la Semana 10.")
        return

    try:
        img = Image.open(f)
        t0 = time.perf_counter()
        gris, mascara, regiones, area_mean, area_std, int_hist, lbp, lbp_hist, features = extraer_caracteristicas_s10(
            img, min_area, radius
        )
        timed("Extracción de características (S10)", t0)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("Regiones Válidas", len(regiones))
        with k2: kpi("Área Media", f"{area_mean:.1f} px")
        with k3: kpi("Desv. Est. Área", f"{area_std:.1f}")
        with k4: kpi("Dimensión Vector", len(features))

        st.markdown("---")
        t_reg, t_hist, t_lbp, t_vec = st.tabs([
            "🔍 Regiones (Otsu)", "📊 Histograma Intensidad", "🌀 Textura LBP", "🧬 Vector Resultante"
        ])

        with t_reg:
            col_a, col_b = st.columns(2)
            col_a.image(gris, caption="Escala de Grises", use_container_width=True)
            col_b.image(mascara.astype(float), caption=f"Máscara (Regiones > {min_area}px)", use_container_width=True)

        with t_hist:
            fig_hist = px.bar(x=np.arange(32), y=int_hist, labels={'x': 'Rangos de Intensidad (Bins)', 'y': 'Frecuencia Normalizada'})
            fig_hist.update_layout(title="Histograma Global de Intensidades (32 Bins)", **PLOT)
            st.plotly_chart(fig_hist, use_container_width=True)

        with t_lbp:
            col_lbp1, col_lbp2 = st.columns(2)
            col_lbp1.image(lbp, caption="Mapa de Patrones Binarios Locales (LBP)", use_container_width=True, clamp=True)
            
            fig_lbp = px.bar(x=np.arange(len(lbp_hist)), y=lbp_hist, labels={'x': 'Patrón LBP (Bins)', 'y': 'Frecuencia'})
            fig_lbp.update_layout(title=f"Histograma de Textura LBP (Radio={radius})", **PLOT)
            col_lbp2.plotly_chart(fig_lbp, use_container_width=True)

        with t_vec:
            fig_vec = px.bar(x=np.arange(len(features)), y=features, color=features, color_continuous_scale="Viridis")
            fig_vec.update_layout(title="Vector Final de Características (Región + Intensidad + LBP)", **PLOT)
            st.plotly_chart(fig_vec, use_container_width=True)
            st.caption(f"Vector de {len(features)} dimensiones guardado exitosamente en `artifacts/semana10_features.npy`")

    except Exception as e:
        st.error(f"Error al extraer características de la imagen: {e}")


def pag_integrado():
    st.title("🚦 Sistema Inteligente de Control Vehicular")
    st.caption("Detector de señales de tránsito · Dashboard del semestre")
    c = st.columns(4)
    regs = leer_sqlite(1000)
    with c[0]: kpi("Señales procesadas", len(regs))
    with c[1]: kpi("Normas indexadas", len(NORMAS))
    with c[2]: kpi("Casos viales", len(CASOS))
    with c[3]: kpi("Clases detectables", len(CLASES))
    f = st.file_uploader("⚡ Pipeline completo: sube una señal", type=["png", "jpg", "jpeg", "webp"], key="full")
    if f:
        try:
            img = Image.open(f)
            t0 = time.perf_counter()
            pid = clasificar_imagen(img, cargar_modelo())
            guardar_sqlite(pid); actualizar_ontologia(pid)
            a, p, col = REGLAS[CLASES[pid]]
            l, r = st.columns([1, 2])
            l.image(img, use_container_width=True)
            r.markdown(f'<div class="glass"><h2>{CLASES[pid]}</h2>{pill(p, col)}<h4>{a}</h4></div>', unsafe_allow_html=True)
            try:
                r.json(buscar_normas(CLASES[pid], 1)[0])
            except Exception:
                pass
            if canny:
                *_, umbral, n = vision(img, 2.0)
                r.caption(f"Visión: Otsu={umbral:.3f} · {n} regiones")
            if local_binary_pattern:
                _, _, _, _, _, _, _, _, features = extraer_caracteristicas_s10(img)
                r.caption(f"Semana 10: Vector LBP/Histogramas extraído ({len(features)} dimensiones guardadas)")
            timed("Pipeline integrado", t0)
        except Exception as e:
            st.error(f"Error en el pipeline: {e}")
    if len(regs):
        fig = px.pie(regs, names="etiqueta_nombre", hole=.55, title="Distribución de señales detectadas",
                     color_discrete_sequence=["#08fdd8", "#7c3aed", "#ffb300", "#ff2e63"])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)


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
}

try:
    ROUTER[pagina]()
except Exception as e:
    st.error(f"Error inesperado en el módulo: {e}")