# Documentación de la Práctica - Semana 04: Marco Tecnológico, Búsqueda A* y Decisión Minimax

## 1. Resumen del Módulo
Durante la Semana 04, se incorporó el **Marco Tecnológico de la Inteligencia Artificial** al **Sistema Inteligente Híbrido de Reconocimiento de Señales de Tránsito**. El objetivo principal de este módulo fue expandir las capacidades del sistema más allá del reconocimiento visual y la inferencia simbólica mediante la implementación de dos pilares fundamentales: la **búsqueda informada mediante el algoritmo A* (A-Star)** para la navegación y planificación de rutas autónomas en entornos viales, y la **toma de decisiones adversariales mediante el algoritmo Minimax** para la evaluación táctica de maniobras seguras frente al tráfico circundante.

## 2. Componentes y Estructura Desarrollada
* **Módulo de Marco Tecnológico y Algoritmos de Búsqueda/Juegos (`src/semana04_marco_tecnologico.py`):**
  * **Planificador de Ruta Autónoma (Algoritmo A*):**
    * Modela la navegación vehicular como un espacio de estados en una cuadrícula vial (`MAPA_VIAL`), considerando zonas transitables y bloqueos/obstáculos.
    * Incorpora una función de prioridad $f(n) = g(n) + h(n)$, donde $g(n)$ representa el costo acumulado por tramo recorrido y $h(n)$ aplica la **Heurística de Manhattan** para estimar la distancia restante hasta el destino objetivo.
  * **Motor de Toma de Decisiones Tácticas (Algoritmo Minimax):**
    * Modela la interacción del vehículo autónomo frente al entorno de tráfico dinámico como un sistema adversarial (MAX / MIN).
    * Evalúa recursivamente los estados posibles del entorno para seleccionar la maniobra óptima (cambio o mantenimiento de carril seguro) que maximice la seguridad y la fluidez del vehículo frente a movimientos de otros actores viales.
* **Integración en la Arquitectura Híbrida (`main.py`):**
  * Conecta de forma secuencial la percepción de señales por visión y reglas (Semana 02), el análisis taxonómico (Semana 03) y la planificación de rutas junto a la toma de decisiones estratégicas (Semana 04).

## 3. Formalización Teórica y Conceptos de Ingeniería

| Componente Formal | Implementación en Búsqueda A* (Ruta Vehicular) | Implementación en Minimax (Decisión de Tráfico) |
| :--- | :--- | :--- |
| **Estado ($s$)** | Coordenadas $(r, c)$ del vehículo en el mapa vial | Configuración actual de los carriles/posiciones viales |
| **Acción ($a$)** | Desplazamiento cardinal (Norte, Sur, Este, Oeste) | Selección de carril o maniobra de evasión / avance |
| **Transición ($T$)** | Movimiento a una celda contigua sin obstáculos (`#`) | Cambio del tablero de tráfico tras el turno del vehículo o entorno |
| **Prueba de Meta** | Alcanzar la coordenada de destino especificada (`META`) | Lograr un estado de trayectoria segura sin colisión/bloqueo |
| **Costo / Heurística** | $g(n)$: tramos recorridos \| $h(n)$: Distancia Manhattan | Función de Utilidad: $+1$ (Maniobra segura), $-1$ (Bloqueo/Riesgo), $0$ (Neutro) |

## 4. Trazabilidad de Resultados y Validación

Los experimentos ejecutados en la consola arrojaron los siguientes resultados reproducibles:

### A. Planificación Vial con A*
* **Coordenadas de Origen y Meta:** Inicio en `(0, 0)` con objetivo en `(4, 4)`.
* **Trazado de Ruta Óptima:** `[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]`.
* **Costo de Desplazamiento:** `8` segmentos viales recorridos de forma óptima esquivando la pared de obstáculos centrales.

### B. Toma de Decisiones con Minimax
* **Estado de Tráfico Evaluado:** `['X', 'O', 'X', 'O', 'X', ' ', ' ', ' ', 'O']`.
* **Carril / Posición Seleccionada:** Posición `6` (maniobra defensiva óptima calculada para el vehículo 'X').
* **Resultado:** Elección de la decisión que maximiza la fluidez y minimiza el riesgo de colisión frente al comportamiento de otros vehículos en la vía.

## 5. Conclusión
La implementación de la Semana 04 dota al sistema de visión y reglas lógicas previos de una capa de **planificación inteligente y autonomía reactiva**. Con la integración de A* y Minimax, el vehículo no solo reconoce señales de tránsito como el PARE o Ceda el Paso, sino que también es capaz de calcular el camino más corto hacia su destino esquivando obstáculos y tomar decisiones tácticas seguras de conducción en tiempo real.