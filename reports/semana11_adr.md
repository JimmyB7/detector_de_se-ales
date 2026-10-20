# ADR Semana 11 - Selección de Modelo

## Contexto
Se evaluaron tres modelos de clasificación para el detector de señales de tránsito bajo un protocolo estricto (mismo dataset y semilla).

## Evidencia
Resultados extraídos de `artifacts/benchmark.csv`:
- **Mejor Modelo**: logistic con Accuracy 0.26 y tiempo 0.0587s.

## Decisión
Se selecciona **logistic** por ofrecer el balance idóneo entre precisión predictiva y eficiencia temporal.
