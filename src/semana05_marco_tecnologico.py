import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
KB_PATH = DATA_DIR / "base_conocimiento.txt"
REPORTS_DIR = ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "semana05.md"

# Definición de reglas heurísticas
RULES = [
    (lambda q: "pare" in q or "stop" in q or "roja" in q, "detener_vehiculo"),
    (lambda q: "ceda" in q or "velocidad" in q or "escolar" in q, "reducir_velocidad"),
    (lambda q: "camara" in q or "sucia" in q or "obstruida" in q, "alerta_mantenimiento_sensor"),
    (lambda q: "lluvia" in q or "mojado" in q, "ajustar_distancia_frenado"),
    (lambda q: "contravia" in q or "direccion" in q, "corregir_trayectoria"),
]

# Datos de entrenamiento para clasificación del tipo de evento
TRAIN_X = [
    "se detecto una senal de pare en la via",
    "frenado inmediato por luz roja en semaforo",
    "detener el auto inmediatamente",
    "reduccion de velocidad por zona escolar",
    "limite de velocidad alcanzado sesenta kilometros",
    "ceda el paso a los vehiculos de la rotonda",
    "lente de la camara con suciedad o empanado",
    "fallo en la resolucion del sensor visual",
    "obstruccion del sensor por polvo",
    "pavimento mojado por lluvia intensa",
    "ajuste de frenado automatico por piso deslizante",
    "girar a la derecha por direccion obligatoria",
    "alerta de contravia en el carril actual",
    "senializacion de curva peligrosa a la izquierda",
    "vehiculo detenido correctamente ante la senal",
]

TRAIN_Y = [
    "seguridad_critica", "seguridad_critica", "seguridad_critica",
    "regulacion_velocidad", "regulacion_velocidad", "regulacion_velocidad",
    "mantenimiento_camara", "mantenimiento_camara", "mantenimiento_camara",
    "seguridad_critica", "seguridad_critica", "informacion_vial",
    "informacion_vial", "informacion_vial", "seguridad_critica",
]


def load_documents() -> list[str]:
    """Carga y valida los registros de la base de conocimiento en texto."""
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {KB_PATH}")

    docs = [line.strip() for line in KB_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(docs) < 8:
        raise ValueError("La base de conocimiento debe contener al menos 8 entradas.")
    return docs


def build_hybrid_system():
    """Inicializa los componentes de extracción TF-IDF y clasificación de eventos."""
    docs = load_documents()

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(docs)

    classifier = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    classifier.fit(TRAIN_X, TRAIN_Y)

    return docs, vectorizer, doc_matrix, classifier


def answer_event(query: str, docs, vectorizer, doc_matrix, classifier) -> dict:
    """Aplica reglas y consulta el modelo TF-IDF para categorizar el evento."""
    q = query.lower()

    fired = [name for condition, name in RULES if condition(q)]

    query_vec = vectorizer.transform([q])
    similarities = cosine_similarity(query_vec, doc_matrix)[0]
    best_index = int(similarities.argmax())

    label = str(classifier.predict([q])[0])

    return {
        "reglas": fired,
        "evidencia": docs[best_index],
        "similitud": float(similarities[best_index]),
        "clase": label,
    }


def write_report(rows: list[tuple[str, dict]]) -> None:
    """Genera el reporte en formato Markdown."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reporte Semana 05 - Sistema Híbrido de Control",
        "",
        "## Evaluaciones Registradas",
        "",
    ]

    for i, (query, result) in enumerate(rows, start=1):
        reglas_str = ", ".join(result["reglas"]) if result["reglas"] else "Ninguna"
        lines.extend([
            f"### Evento {i}: \"{query}\"",
            f"- **Reglas asociadas:** `{reglas_str}`",
            f"- **Norma / Coincidencia:** {result['evidencia']}",
            f"- **Similitud TF-IDF:** `{result['similitud']:.3f}`",
            f"- **Categoría asignada:** `{result['clase']}`",
            "",
        ])

    lines.extend([
        "## Observaciones Técnicas",
        "- El uso de TF-IDF permite relacionar términos de entrada con las reglas definidas en la base de conocimiento.",
        "- Las reglas directas garantizan prioridad sobre la salida del modelo estadístico en casos críticos.",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def ejecutar_semana_05():
    docs, vectorizer, doc_matrix, classifier = build_hybrid_system()

    test_queries = [
        "Se vislumbra una senal de PARE en la interseccion",
        "Atencion la camara esta sucia y obstruida por polvo",
        "Reducir la marcha en la zona escolar por limite de velocidad",
    ]

    rows = []
    print("--- Evaluación Semana 05 ---")
    for query in test_queries:
        res = answer_event(query, docs, vectorizer, doc_matrix, classifier)
        rows.append((query, res))
        print(f"\nEntrada: {query}")
        print(json.dumps(res, indent=2, ensure_ascii=False))

    write_report(rows)
    print(f"\nReporte guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    ejecutar_semana_05()