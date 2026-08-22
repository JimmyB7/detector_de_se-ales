"""
Orquestador Principal - Sistema Inteligente Híbrido Aplicado.
Proyecto: Análisis y reconocimiento de señales de tránsito con IA y acciones de control.
Estandarizado para Python 3.13.x.
"""

from pathlib import Path
import sys

# Asegurar que la ruta 'src' sea reconocida por Python
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from semana02_fundamentos import model, motor_de_reglas_vehicular, X_test, y_test
from semana03_taxonomia import classify_problem, read_cases

def ejecutar_sistema_hibrido():
    print("=" * 80)
    print("SISTEMA HÍBRIDO INTELIGENTE DE SEÑALES DE TRÁNSITO")
    print("=" * 80)
    
    # --- FASE 1: Visión Artificial y Motor de Reglas (Semana 02) ---
    print("\n[FASE 1] Evaluación del Modelo Predictivo y Control Vehicular:")
    predicciones = model.predict(X_test[:3])
    for i, pred in enumerate(predicciones):
        regla = motor_de_reglas_vehicular(pred)
        print(f"  -> Fotograma #{i+1} | Señal Detectada: {regla['senial']}")
        print(f"     ↳ Acción del Sistema: {regla['accion']}")
        print(f"     ↳ Prioridad de Respuesta: {regla['prioridad']}")

    # --- FASE 2: Clasificación Taxonómica y Análisis de Casos (Semana 03) ---
    print("\n[FASE 2] Análisis Taxonómico de Casos de Estudio:")
    try:
        cases = read_cases()
        print(f"  -> Total de casos cargados desde CSV: {len(cases)}")
        if cases:
            # Analizamos de ejemplo el primer caso del conjunto
            ejemplo = cases[0]
            primary, detected, scores = classify_problem(ejemplo)
            print(f"  -> Caso de prueba: '{ejemplo[:55]}...'")
            print(f"     ↳ Categoría Principal: {primary}")
            print(f"     ↳ Áreas de IA Detectadas: {', '.join(detected)}")
    except FileNotFoundError as e:
        print(f"  -> Advertencia: {e}")

    print("\n" + "=" * 80)
    print("FLUJO HÍBRIDO EJECUTADO CORRECTAMENTE")
    print("=" * 80)

if __name__ == "__main__":
    ejecutar_sistema_hibrido()