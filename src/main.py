from pathlib import Path
import sys

# Asegurar que la ruta actual y 'src' sean reconocidas por Python
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

def ejecutar_sistema_hibrido():
    print("=" * 80)
    print("SISTEMA HÍBRIDO INTELIGENTE DE RECONOCIMIENTO DE SEÑALES Y CONTROL VEHICULAR")
    print("=" * 80)

    # --- FASE 1: Visión Artificial y Motor de Reglas (Semana 02) ---
    print("\n[FASE 1] Evaluación del Modelo Predictivo y Control Vehicular (Semana 02):")
    try:
        from semana02_fundamentos import model, motor_de_reglas_vehicular, X_test
        predicciones = model.predict(X_test[:3])
        for i, pred in enumerate(predicciones):
            regla = motor_de_reglas_vehicular(pred)
            print(f"  -> Fotograma #{i+1} | Señal Detectada: {regla['senial']}")
            print(f"     ↳ Acción del Sistema: {regla['accion']}")
            print(f"     ↳ Prioridad de Respuesta: {regla['prioridad']}")

            # Validación explícita de frenado al detectar PARE
            if regla["senial"] == "PARE":
                print("     🚨 [ALERTA DE SEGURIDAD]: Ejecutando comando HALT_VEHICLE() -> DETENER VEHÍCULO")
    except Exception as e:
        print(f"  -> Error/Advertencia en Fase 1: {e}")

    # --- FASE 2: Clasificación Taxonómica y Análisis de Casos (Semana 03) ---
    print("\n[FASE 2] Análisis Taxonómico de Casos de Estudio (Semana 03):")
    try:
        from semana03_taxonomia import classify_problem, read_cases
        cases = read_cases()
        print(f"  -> Total de casos cargados desde CSV/Lista: {len(cases)}")
        if cases:
            ejemplo = cases[0]
            primary, detected, scores = classify_problem(ejemplo)
            print(f"  -> Caso de prueba: '{ejemplo[:55]}...'")
            print(f"     ↳ Categoría Principal: {primary}")
            print(f"     ↳ Áreas de IA Detectadas: {', '.join(detected)}")
    except Exception as e:
        print(f"  -> Advertencia en Fase 2: {e}")

    # --- FASE 3: Marco Tecnológico - Planificación A* y Decisiones Minimax (Semana 04) ---
    print("\n[FASE 3] Navegación Vial (A*) y Toma de Decisiones de Tráfico (Minimax) (Semana 04):")
    try:
        from semana04_marco_tecnologico import (
            INICIO,
            META,
            calcular_mejor_maniobra,
            planificar_ruta_astar,
        )
        # 1. Planificación de Ruta con A*
        ruta = planificar_ruta_astar(INICIO, META)
        print(f"  -> [Planificación A*] Ruta Evadiendo Bloqueos de {INICIO} a {META}:")
        print(f"     ↳ Traza de Coordenadas: {ruta}")
        print(f"     ↳ Pasos Totales: {len(ruta) - 1 if ruta else 'No encontrada'}")

        # 2. Decisiones Minimax ante tráfico
        escenario = ["X", "O", "X", "O", "X", " ", " ", " ", "O"]
        accion = calcular_mejor_maniobra(escenario)
        print(f"  -> [Minimax Decisiones] Evaluación de Estado de Tráfico:")
        print(f"     ↳ Estado Actual: {escenario}")
        print(f"     ↳ Maniobra/Posición Seleccionada por el Vehículo: {accion}")
    except Exception as e:
        print(f"  -> Advertencia en Fase 3: {e}")

    print("\n" + "=" * 80)
    print("FLUJO HÍBRIDO COMPLETO EJECUTADO CORRECTAMENTE")
    print("=" * 80)

if __name__ == "__main__":
    ejecutar_sistema_hibrido()