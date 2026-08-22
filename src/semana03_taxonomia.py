"""
Módulo de Taxonomía de Inteligencia Artificial.
Proyecto: Análisis de señales de tránsito mediante IA con cámara.
Estandarizado para Python 3.13.x.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import re
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "data" / "casos_ia.csv"
REPORT_FILE = ROOT / "reports" / "semana03.md"

@dataclass(frozen=True)
class Category:
    name: str
    keywords: tuple[str, ...]

CATEGORIES = [
    Category("Visión por computador", (
        "imagen", "imagenes", "foto", "fotografia", "camara",
        "rostro", "rostros", "peaton", "peatones", "senal", "senales", "fotogramas", "lente"
    )),
    Category("Procesamiento de lenguaje natural", (
        "texto", "comentario", "comentarios", "correo", "correos", "chatbot",
        "contrato", "contratos", "nombres", "lenguaje", "asistente virtual"
    )),
    Category("Aprendizaje automático predictivo", (
        "predecir", "probabilidad", "demanda", "fraude", "fraudes", "sensores", "confianza"
    )),
    Category("Sistemas de recomendación", (
        "recomendar", "preferencias", "historial", "sugerir", "actualizaciones"
    )),
    Category("Búsqueda y optimización", (
        "ruta", "rutas", "horario", "horarios", "combinacion optima",
        "optimizar", "capacidad maxima", "resolucion"
    )),
    Category("Sistemas expertos", (
        "diagnostico", "diagnosticos", "reglas", "politicas", "solicitud", "frenado", "alerta"
    )),
    Category("Robótica y sistemas autónomos", (
        "robot", "robots", "dron", "drones", "vehiculo", "obstaculos", "embebido"
    )),
]

# Reglas personalizadas adaptadas al dominio de señales de tránsito y visión artificial
CUSTOM_RULES = {
    "Visión por computador": ("fotogramas", "lente", "video"),
    "Aprendizaje automático predictivo": ("confianza",),
    "Búsqueda y optimización": ("resolucion", "latencias"),
    "Sistemas expertos": ("frenado", "alerta"),
    "Robótica y sistemas autónomos": ("vehiculo", "embebido"),
}

# Referencia manual correspondiente a los 20 casos del CSV de señales de tránsito
MANUAL_REFERENCE = [
    "Visión por computador",
    "Visión por computador",
    "Aprendizaje automático predictivo",
    "Búsqueda y optimización",
    "Sistemas de recomendación",
    "Visión por computador",
    "Visión por computador",
    "Procesamiento de lenguaje natural",
    "Aprendizaje automático predictivo",
    "Sistemas expertos",
    "Visión por computador",
    "Visión por computador",
    "Robótica y sistemas autónomos",
    "Sistemas expertos",
    "Visión por computador",
    "Visión por computador",
    "Visión por computador",
    "Sistemas expertos",
    "Búsqueda y optimización",
    "Búsqueda y optimización",
]

def normalize(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return re.sub(r"\s+", "", text).strip()

def normalize_header(text: str) -> str:
    return normalize(text).replace("descripcion", "")

def contains_keyword(text: str, keyword: str) -> bool:
    normalized_text = f" {normalize(text)} "
    normalized_keyword = normalize(keyword)
    return f"{normalized_keyword}" in normalized_text

def build_categories() -> list[Category]:
    result = []
    for category in CATEGORIES:
        extra = CUSTOM_RULES.get(category.name, ())
        result.append(Category(category.name, category.keywords + tuple(extra)))
    return result

def classify_problem(text: str) -> tuple[str, list[str], dict[str, int]]:
    scores = {}
    for category in build_categories():
        score = sum(contains_keyword(text, keyword) for keyword in category.keywords)
        scores[category.name] = score
    
    matches = [
        (score, index, category.name)
        for index, category in enumerate(build_categories())
        if (score := scores[category.name]) > 0
    ]
    matches.sort(key=lambda item: (-item[0], item[1]))
    detected = [name for _, _, name in matches]
    primary = detected[0] if detected else "Requiere análisis"
    return primary, detected or ["Requiere análisis"], scores

def read_cases() -> list[str]:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"No existe {CSV_FILE}. Crea data/casos_ia.csv antes de ejecutar la práctica.")
    
    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("El CSV está vacío o no contiene encabezados.")
        
        original_headers = list(reader.fieldnames)
        reader.fieldnames = [normalize_header(name) for name in reader.fieldnames]
        
        if "descripcion" not in reader.fieldnames:
            raise ValueError(f"No se encontró la columna 'descripcion'. Encabezados encontrados: {original_headers}")
        
        cases = []
        for row in reader:
            description = (row.get("descripcion") or "").strip()
            if description:
                cases.append(description)
                
        if len(cases) < 20:
            raise ValueError(f"La práctica requiere al menos 20 casos y el archivo contiene {len(cases)}.")
        return cases

def write_report(results: list[dict]) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    reference_count = min(len(results), len(MANUAL_REFERENCE))
    matches = sum(
        1 for i in range(reference_count) if results[i]["primary"] == MANUAL_REFERENCE[i]
    )
    accuracy = 100 * matches / reference_count if reference_count else 0.0

    lines = [
        "# Semana 03: Taxonomía de Inteligencia Artificial",
        "## Resultado automático frente a clasificación manual de referencia",
        "| Caso | Categoría automática principal | Categorías detectadas | Manual | Estado |",
        "|---|---|---|---|---|",
    ]
    for i, result in enumerate(results, start=1):
        manual = MANUAL_REFERENCE[i - 1] if i - 1 < len(MANUAL_REFERENCE) else "Pendiente"
        status = "Coincide" if result["primary"] == manual else "Revisar"
        detected = ", ".join(result["detected"])
        lines.append(f"| {i} | {result['primary']} | {detected} | {manual} | {status} |")

    lines += [
        f"\nCoincidencia con la referencia: **{accuracy:.2f}%** ({matches}/{reference_count}).",
        "## Cinco reglas propias",
        "Se ampliaron palabras clave en `CUSTOM_RULES` para capturar terminología específica de captura de video vehicular, umbrales de confianza, latencias de procesamiento y respuestas de frenado en sistemas embebidos de tráfico.",
        "## Discrepancias y análisis",
        "Se evaluaron los casos donde convergen la visión artificial y los sistemas expertos (por ejemplo, detectar la señal y activar alertas de frenado), priorizando el núcleo perceptivo o de control según el objetivo principal del requerimiento.",
        "## Nota técnica",
        "Un problema real en sistemas híbridos pertenece a varias áreas de IA. La columna 'principal' usa la categoría con mayor cantidad de coincidencias.",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    cases = read_cases()
    results = []
    print("=" * 80)
    print("SEMANA 03: TAXONOMÍA DE INTELIGENCIA ARTIFICIAL")
    print("=" * 80)
    for i, case in enumerate(cases, start=1):
        primary, detected, scores = classify_problem(case)
        results.append({
            "description": case,
            "primary": primary,
            "detected": detected,
            "scores": scores,
        })
        print(f"{i:02d}. {case}")
        print(f"   Principal: {primary}")
        print(f"   Áreas detectadas: {', '.join(detected)}")
    
    write_report(results)
    print(f"\nCasos procesados: {len(results)}")
    print(f"Reporte generado: {REPORT_FILE}")

if __name__ == "__main__":
    main()