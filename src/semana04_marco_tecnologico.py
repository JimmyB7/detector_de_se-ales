import heapq
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Definición del entorno gráfico (0,0 a 4,4)
MAPA_VIAL = [
    ".....",
    ".###.",
    "...#.",
    ".#...",
    ".....",
]
INICIO = (0, 0)
META = (4, 4)

LINEAS_VICTORIA = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
)


def heuristica_manhattan(a: tuple, b: tuple) -> int:
    """Distancia Manhattan entre dos puntos de la grilla."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def obtener_vecinos(nodo: tuple):
    """Retorna coordenadas adyacentes válidas sin obstáculos."""
    r, c = nodo
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(MAPA_VIAL) and 0 <= nc < len(MAPA_VIAL[0]):
            if MAPA_VIAL[nr][nc] != "#":
                yield (nr, nc)


def planificar_ruta_astar(inicio: tuple, meta: tuple) -> list[tuple] | None:
    """Encuentra el camino óptimo evadiendo obstáculos usando A*."""
    frontera = [(0, inicio)]
    procedencia = {inicio: None}
    costo_acumulado = {inicio: 0}

    while frontera:
        _, actual = heapq.heappop(frontera)

        if actual == meta:
            break

        for siguiente in obtener_vecinos(actual):
            nuevo_costo = costo_acumulado[actual] + 1
            if siguiente not in costo_acumulado or nuevo_costo < costo_acumulado[siguiente]:
                costo_acumulado[siguiente] = nuevo_costo
                prioridad = nuevo_costo + heuristica_manhattan(siguiente, meta)
                heapq.heappush(frontera, (prioridad, siguiente))
                procedencia[siguiente] = actual

    if meta not in procedencia:
        return None

    camino = []
    actual = meta
    while actual is not None:
        camino.append(actual)
        actual = procedencia[actual]
    return list(reversed(camino))


def evaluar_estado_trafico(escenario: list) -> str | None:
    """Determina si un estado del tablero representa una alineación ganadora."""
    for a, b, c in LINEAS_VICTORIA:
        if escenario[a] == escenario[b] == escenario[c] and escenario[a] != " ":
            return escenario[a]
    return None


def minimax_maniobra(escenario: list, es_maximizando: bool) -> int:
    """Aplica algoritmo Minimax para evaluar estados de maniobra."""
    ganador = evaluar_estado_trafico(escenario)
    if ganador == "X":
        return 1
    if ganador == "O":
        return -1
    if " " not in escenario:
        return 0

    scores = []
    marca = "X" if es_maximizando else "O"
    for i, celda in enumerate(escenario):
        if celda == " ":
            siguiente = escenario.copy()
            siguiente[i] = marca
            scores.append(minimax_maniobra(siguiente, not es_maximizando))

    return max(scores) if es_maximizando else min(scores)


def calcular_mejor_maniobra(escenario: list) -> int:
    """Selecciona el índice de movimiento con mayor puntaje según Minimax."""
    opciones = []
    for i, celda in enumerate(escenario):
        if celda == " ":
            siguiente = escenario.copy()
            siguiente[i] = "X"
            opciones.append((minimax_maniobra(siguiente, False), i))
    return max(opciones)[1] if opciones else -1


def ejecutar_semana_04():
    print("--- Planificación de Ruta con A* ---")
    ruta = planificar_ruta_astar(INICIO, META)
    print(f"Origen: {INICIO} | Destino: {META}")
    print(f"Ruta calculada: {ruta}")

    print("\n--- Evaluación de Maniobra con Minimax ---")
    escenario_trafico = ["X", "O", "X", "O", "X", " ", " ", " ", "O"]
    posicion = calcular_mejor_maniobra(escenario_trafico)
    print(f"Estado inicial: {escenario_trafico}")
    print(f"Movimiento óptimo seleccionado: {posicion}")


if __name__ == "__main__":
    ejecutar_semana_04()