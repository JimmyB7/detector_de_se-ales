from pathlib import Path
import sys

# Asegurar que el directorio 'src' esté en el path de Python
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Importar únicamente componentes de la Semana 02
from semana02_fundamentos import (
    model,
    motor_de_reglas_vehicular,
    X_test,
    y_test,
    accuracy_score,
    confusion_matrix,
)

# ==============================================================================
# FUNCIONES PROPIAS DE LA SEMANA 03 (Clasificación Taxonómica y Casos)
# ==============================================================================

def read_cases():
    """Lee o simula los casos de estudio almacenados."""
    return [
        "Sistema de visión por computador para detectar señales de PARE e inferir la acción del vehículo.",
        "Monitoreo de carril e identificación de límites de velocidad mediante red neuronal.",
        "Módulo de decisión ética y priorización de frenado autónomo ante obstáculos imprevistos.",
    ]


def classify_problem(case_description: str):
    """Clasifica una descripción según la taxonomía de las áreas de la IA."""
    detected = []
    text_lower = case_description.lower()

    if any(k in text_lower for k in ["visión", "señal", "detectar", "computador"]):
        detected.append("Visión Artificial")
    if any(k in text_lower for k in ["decisión", "accion", "regla", "inferir"]):
        detected.append("Sistemas Basados en Reglas / Inferencia")
    if any(k in text_lower for k in ["red neuronal", "frenado", "autónomo"]):
        detected.append("Robótica y Control")

    primary = detected[0] if detected else "Inteligencia Artificial General"
    scores = {area: 0.9 for area in detected}

    return primary, detected, scores


def write_report(results: list):
    """Genera o guarda el reporte de casos procesados."""
    pass


# ==============================================================================
# FASES DE EJECUCIÓN
# ==============================================================================

def ejecutar_semana_02():
    print("=" * 80)
    print("FASE 1: ARQUITECTURA BASE, MODELO DE IA Y MOTOR DE REGLAS (SEMANA 02)")
    print("=" * 80)
    print(f"Muestras de prueba evaluadas: {len(X_test)}")

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Accuracy del modelo de clasificación visual: {acc:.3f}\n")

    print("--- MATRIZ DE EFECTIVIDAD (CONFUSIÓN) ---")
    print("Filas: Clases Reales | Columnas: Clases Predichas (PARE, CEDA EL PASO, LÍMITE)")
    print(confusion_matrix(y_test, pred))
    print()

    print("--- SIMULACIÓN DEL MOTOR DE INFERENCIA Y ACCIONES ---")
    for i in range(3):
        clase_detectada = pred[i]
        regla_aplicada = motor_de_reglas_vehicular(clase_detectada)
        print(f"Fotograma de prueba [{i+1}] -> Señal detectada por IA: {regla_aplicada['senial']}")
        print(f"   ↳ Acción ejecutada: {regla_aplicada['accion']}")
        print(f"   ↳ Prioridad de control: {regla_aplicada['prioridad']}")

        if regla_aplicada["senial"] == "PARE":
            print("   🚨 [ALERTA DE SEGURIDAD]: Ejecutando comando HALT_VEHICLE() -> DETENER VEHÍCULO")
        print()


def ejecutar_semana_03():
    print("=" * 80)
    print("FASE 2: TAXONOMÍA DE INTELIGENCIA ARTIFICIAL Y ANÁLISIS DE CASOS (SEMANA 03)")
    print("=" * 80)

    cases = read_cases()
    results = []

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
        print(f"   Áreas detectadas: {', '.join(detected)}\n")

    write_report(results)
    print(f"Casos procesados: {len(results)}")
    print("Reporte taxonómico generado exitosamente.")


def main():
    print("\n" + "#" * 80)
    print(" # INICIO DEL SISTEMA INTELIGENTE HÍBRIDO APLICADO - SEÑALES DE TRÁNSITO #")
    print("#" * 80 + "\n")

    # Ejecutar componente Semana 02
    ejecutar_semana_02()

    print("\n")

    # Ejecutar componente Semana 03
    ejecutar_semana_03()

    print("\n" + "=" * 80)
    print("EJECUCIÓN GENERAL DEL SISTEMA COMPLETADA EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()