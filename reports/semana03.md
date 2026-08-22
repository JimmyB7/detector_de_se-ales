# Semana 03: Taxonomía de Inteligencia Artificial
## Resultado automático frente a clasificación manual de referencia
| Caso | Categoría automática principal | Categorías detectadas | Manual | Estado |
|---|---|---|---|---|
| 1 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 2 | Requiere análisis | Requiere análisis | Visión por computador | Revisar |
| 3 | Aprendizaje automático predictivo | Aprendizaje automático predictivo, Visión por computador, Sistemas de recomendación | Aprendizaje automático predictivo | Coincide |
| 4 | Visión por computador | Visión por computador, Búsqueda y optimización | Búsqueda y optimización | Revisar |
| 5 | Sistemas de recomendación | Sistemas de recomendación | Sistemas de recomendación | Coincide |
| 6 | Visión por computador | Visión por computador, Robótica y sistemas autónomos | Visión por computador | Coincide |
| 7 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 8 | Visión por computador | Visión por computador, Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Revisar |
| 9 | Visión por computador | Visión por computador, Aprendizaje automático predictivo | Aprendizaje automático predictivo | Revisar |
| 10 | Sistemas expertos | Sistemas expertos, Visión por computador | Sistemas expertos | Coincide |
| 11 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 12 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 13 | Robótica y sistemas autónomos | Robótica y sistemas autónomos, Visión por computador, Búsqueda y optimización | Robótica y sistemas autónomos | Coincide |
| 14 | Sistemas expertos | Sistemas expertos, Visión por computador | Sistemas expertos | Coincide |
| 15 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 16 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 17 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 18 | Robótica y sistemas autónomos | Robótica y sistemas autónomos, Visión por computador, Sistemas expertos | Sistemas expertos | Revisar |
| 19 | Búsqueda y optimización | Búsqueda y optimización, Visión por computador | Búsqueda y optimización | Coincide |
| 20 | Búsqueda y optimización | Búsqueda y optimización | Búsqueda y optimización | Coincide |

Coincidencia con la referencia: **75.00%** (15/20).
## Cinco reglas propias
Se ampliaron palabras clave en `CUSTOM_RULES` para capturar terminología específica de captura de video vehicular, umbrales de confianza, latencias de procesamiento y respuestas de frenado en sistemas embebidos de tráfico.
## Discrepancias y análisis
Se evaluaron los casos donde convergen la visión artificial y los sistemas expertos (por ejemplo, detectar la señal y activar alertas de frenado), priorizando el núcleo perceptivo o de control según el objetivo principal del requerimiento.
## Nota técnica
Un problema real en sistemas híbridos pertenece a varias áreas de IA. La columna 'principal' usa la categoría con mayor cantidad de coincidencias.