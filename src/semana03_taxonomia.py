from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from semana02_fundamentos import (
    accuracy_score,
    confusion_matrix,
    model,
    motor_de_reglas_vehicular,
    X_test,
    y_test,
)


def read_cases() -> list[str]:
    """Carga los casos de prueba para clasificación del dominio."""
    return [
        "Sistema de visión por computador para detectar señales de PARE e inferir la acción del vehículo.",
        "Monitoreo de carril e identificación de límites de velocidad mediante red neuronal.",
        "Módulo de decisión ética y priorización de frenado autónomo ante obstáculos imprevistos.",
    ]


def classify_problem(case_description: str) -> tuple[str, list[str], dict]:
    """Clasifica el caso dentro de las áreas funcionales de IA."""
    detected = []
    text_lower = case_description.lower()

    if any(k in text_lower for k in ["visión", "señal", "detectar", "computador"]):
        detected.append("Visión Artificial")
    if any(k in text_lower for k in ["decisión", "accion", "regla", "inferir"]):
        detected.append("Sistemas Basados en Reglas")
    if any(k in text_lower for k in ["red neuronal", "frenado", "autónomo"]):
        detected.append("Robótica y Control")

    primary = detected[0] if detected else "General"
    scores = {area: 0.9 for area in detected}

    return primary, detected, scores


def ejecutar_semana_02():
    print("--- Evaluación Modelo Semana 02 ---")
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Accuracy: {acc:.3f}\n")

    print("Matriz de confusión:")
    print(confusion_matrix(y_test, pred))
    print()

    for i in range(3):
        regla = motor_de_reglas_vehicular(pred[i])
        print(f"Muestra {i+1}: {regla['senial']} -> {regla['accion']}")


def ejecutar_semana_03():
    print("\n--- Análisis Taxonómico Semana 03 ---")
    cases = read_cases()
    for i, case in enumerate(cases, start=1):
        primary, detected, _ = classify_problem(case)
        print(f"{i}. {case}")
        print(f"   Categoría principal: {primary}")
        print(f"   Áreas: {', '.join(detected)}\n")


def main():
    ejecutar_semana_02()
    ejecutar_semana_03()


if __name__ == "__main__":
    main()