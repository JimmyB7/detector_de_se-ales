"""
Orquestador Principal - Sistema Inteligente Híbrido Aplicado.
Proyecto: Análisis y reconocimiento de señales de tránsito con IA y acciones de control.
Estandarizado para Python 3.13.x.
"""

from pathlib import Path
import sys

# Asegurar que el directorio 'src' esté en el path de Python
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Importar componentes de la Semana 02 y Semana 03
from semana02_fundamentos import model, motor_de_reglas_vehicular, X_test, y_test, accuracy_score, confusion_matrix
from semana03_taxonomia import read_cases, classify_problem, write_report


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
        print(f"   ↳ Prioridad de control: {regla_aplicada['prioridad']}\n")


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
        print(f"   Áreas detectadas: {', '.join(detected)}")
    
    write_report(results)
    print(f"\nCasos procesados: {len(results)}")
    print(f"Reporte taxonómico generado exitosamente.")


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