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
from semana03_taxonomia import classify_problem, read_cases
from semana04_marco_tecnologico import (
    INICIO,
    META,
    calcular_mejor_maniobra,
    planificar_ruta_astar,
)
from semana05_marco_tecnologico import ejecutar_semana_05


def main():
    print("==========================================================================")
    print("     SISTEMA DE RECONOCIMIENTO DE SEÑALES Y CONTROL VEHICULAR")
    print("==========================================================================")

    # --- FASE 1: Modelo de Clasificación y Reglas (Semana 02) ---
    print("\n[Fase 1] Evaluación de Clasificación Visual")
    pred_completa = model.predict(X_test)
    acc = accuracy_score(y_test, pred_completa)

    print(f"Accuracy del modelo: {acc:.3f}")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, pred_completa))

    print("\nAcciones asignadas en fotogramas de prueba:")
    for i in range(3):
        regla = motor_de_reglas_vehicular(pred_completa[i])
        print(f"  Fotograma {i+1}: {regla['senial']} -> {regla['accion']}")

    # --- FASE 2: Análisis Taxonómico (Semana 03) ---
    print("\n[Fase 2] Análisis de Casos del Dominio")
    cases = read_cases()
    if cases:
        ejemplo = cases[0]
        primary, detected, _ = classify_problem(ejemplo)
        print(f"  Caso analizado: '{ejemplo[:60]}...'")
        print(f"  Categoría principal: {primary}")
        print(f"  Áreas asociadas: {', '.join(detected)}")

    # --- FASE 3: Planificación y Algoritmos de Búsqueda (Semana 04) ---
    print("\n[Fase 3] Planificación de Ruta (A*) y Evaluación (Minimax)")
    ruta = planificar_ruta_astar(INICIO, META)
    print(f"  Ruta trazada (A*): {ruta}")

    escenario = ["X", "O", "X", "O", "X", " ", " ", " ", "O"]
    posicion = calcular_mejor_maniobra(escenario)
    print(f"  Maniobra seleccionada (Minimax): Posición {posicion}")

    # --- FASE 4: Sistema Híbrido y Recuperación de Información (Semana 05) ---
    print("\n[Fase 4] Sistema Híbrido y Base de Conocimiento")
    ejecutar_semana_05()

    print("\nEjecución finalizada correctamente.")


if __name__ == "__main__":
    main()