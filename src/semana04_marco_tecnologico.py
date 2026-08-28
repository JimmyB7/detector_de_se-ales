"""
SEMANA 04: MARCO TECNOLÓGICO DE LA INTELIGENCIA ARTIFICIAL
Búsqueda de Rutas Vehiculares (A*) y Planificación Adversarial de Maniobras (Minimax)
Aplicado al Sistema Autónomo de Reconocimiento de Señales de Tránsito.
"""

import heapq
from pathlib import Path
import sys

# Asegurar importación de semanas previas si es necesario
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# ==============================================================================
# 1. BÚSQUEDA A* (A-STAR) PARA NAVEGACIÓN Y PLANIFICACIÓN DE RUTA VEHICULAR
# ==============================================================================

# Mapa vial/cuadrícula donde '.' es vía libre y '#' representa un obstáculo/bloqueo
MAPA_VIAL = [
    ".....",
    ".###.",
    "...#.",
    ".#...",
    ".....",
]
INICIO, META = (0, 0), (4, 4)

def heuristica_manhattan(a: tuple, b: tuple) -> int:
    """Estimación heurística de costo restante desde la posición actual a la meta."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def obtener_vecinos_viales(nodo: tuple):
    """Obtiene celdas transitables contiguas para el vehículo (Norte, Sur, Este, Oeste)."""
    r, c = nodo
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(MAPA_VIAL) and 0 <= nc < len(MAPA_VIAL[0]):
            if MAPA_VIAL[nr][nc] != '#':
                yield (nr, nc)

def planificar_ruta_astar(inicio: tuple, meta: tuple):
    """Calcula la ruta vehicular de costo mínimo evadiendo obstáculos usando A*."""
    frontera = [(0, inicio)]
    procedencia = {inicio: None}
    costo_acumulado = {inicio: 0}

    while frontera:
        _, actual = heapq.heappop(frontera)

        if actual == meta:
            break

        for siguiente in obtener_vecinos_viales(actual):
            nuevo_costo = costo_acumulado[actual] + 1
            if siguiente not in costo_acumulado or nuevo_costo < costo_acumulado[siguiente]:
                costo_acumulado[siguiente] = nuevo_costo
                prioridad = nuevo_costo + heuristica_manhattan(siguiente, meta)
                heapq.heappush(frontera, (prioridad, siguiente))
                procedencia[siguiente] = actual

    if meta not in procedencia:
        return None

    # Reconstrucción del camino
    camino, cur = [], meta
    while cur is not None:
        camino.append(cur)
        cur = procedencia[cur]
    return list(reversed(camino))

# ==============================================================================
# 2. DECISIÓN DE MANIOBRA VEHICULAR ADVERSARIAL / MINIMAX
# ==============================================================================
# Modelado de decisiones de conducción competitiva (e.g. evaluación de carril seguro)

LINEAS_VICTORIA = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))

def evaluar_estado_trafico(escenario: list):
    """Evalúa si una maniobra fue segura (X) o un bloqueo/riesgo (O)."""
    for a, b, c in LINEAS_VICTORIA:
        if escenario[a] == escenario[b] == escenario[c] and escenario[a] != " ":
            return escenario[a]
    return None

def minimax_maniobra(escenario: list, es_maximizando: bool):
    """Algoritmo Minimax para predecir y calcular la maniobra vehicular óptima."""
    ganador = evaluar_estado_trafico(escenario)
    if ganador == "X": return 1    # Maniobra segura / Avance óptimo
    if ganador == "O": return -1   # Bloqueo o maniobra de riesgo
    if " " not in escenario: return 0  # Estado de tráfico neutro

    scores = []
    marca = "X" if es_maximizando else "O"
    for i, celda in enumerate(escenario):
        if celda == " ":
            siguiente = escenario.copy()
            siguiente[i] = marca
            scores.append(minimax_maniobra(siguiente, not es_maximizando))

    return max(scores) if es_maximizando else min(scores)

def calcular_mejor_maniobra(escenario: list) -> int:
    """Retorna la mejor acción/celda defensiva para el vehículo autónomo."""
    opciones = []
    for i, celda in enumerate(escenario):
        if celda == " ":
            siguiente = escenario.copy()
            siguiente[i] = "X"
            opciones.append((minimax_maniobra(siguiente, False), i))
    return max(opciones)[1] if opciones else -1

# ==============================================================================
# PRUEBAS DEL MÓDULO SEMANA 04
# ==============================================================================
def ejecutar_semana_04():
    print("=" * 80)
    print("FASE 3: MARCO TECNOLÓGICO, BÚSQUEDA A* Y DECISIÓN MINIMAX (SEMANA 04)")
    print("=" * 80)
    
    # 1. Prueba de Planificación A*
    print("\n--- [PLANIFICACIÓN VIAL CON A*] ---")
    ruta = planificar_ruta_astar(INICIO, META)
    print(f"  -> Punto de Origen: {INICIO} | Destino Objetivo: {META}")
    print(f"  -> Trazado de Ruta Óptima: {ruta}")
    print(f"  -> Costo de Desplazamiento (pasos/segmentos): {len(ruta) - 1 if ruta else 'Inalcanzable'}")

    # 2. Prueba de Decisión Minimax
    print("\n--- [TOMA DE DECISIONES DE TRÁFICO CON MINIMAX] ---")
    escenario_trafico = ["X", "O", "X", "O", "X", " ", " ", " ", "O"]
    posicion_optima = calcular_mejor_maniobra(escenario_trafico)
    print(f"  -> Estado Actual del Entorno: {escenario_trafico}")
    print(f"  -> Posición/Carril Óptimo para la Maniobra Autónoma ('X'): {posicion_optima}")
    print("  -> Estrategia: Maximizar la fluidez vehicular anticipando maniobras del tráfico.")

if __name__ == "__main__":
    ejecutar_semana_04()