# ADR Semana 11 - Selección de Modelo

## Contexto
Se evaluaron tres modelos de clasificación para el detector de señales de tránsito bajo un protocolo estricto (mismo dataset, partición y semilla). El dataset local se carga desde `data/img`; si no contiene suficientes imágenes de las cuatro clases, se utiliza un conjunto sintético de respaldo.

## Evidencia
Resultados extraídos de `artifacts/benchmark.csv`:
- **Mejor Modelo**: logistic con Accuracy 0.3333 y tiempo 0.0369s.

## Decisión
Se selecciona **logistic** por ofrecer el balance idóneo entre precisión predictiva y eficiencia temporal.
