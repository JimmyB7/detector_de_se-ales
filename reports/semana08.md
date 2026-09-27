# Reporte Semana 08: Representaciones del Reconocimiento en Señales de Tránsito

## 1. Integración de Componentes
En esta semana se conectaron tres pilares fundamentales para garantizar la confiabilidad del sistema:
* **Red Neuronal (MLPClassifier):** Encargada del aprendizaje de patrones y la clasificación de parches de imágenes.
* **Base de Datos SQLite (`imagenes.db`):** Registro de metadatos (ID de imagen, etiqueta real y partición del dataset) para la auditoría y trazabilidad de la evidencia.
* **Ontología GraphML (`ontologia.graphml`):** Grafo semántico que dota de significado a las predicciones del modelo dentro del dominio del tránsito.

## 2. Estructura Ontológica del Dominio Vial
Se construyó un grafo dirigido con las siguientes relaciones clave:
1. `detector_mlp` $\rightarrow$ `clasifica` $\rightarrow$ `senal_transito`
2. `senal_transito` $\rightarrow$ `es_tipo` $\rightarrow$ `reglamentaria`
3. `reglamentaria` $\rightarrow$ `contiene_clase` $\rightarrow$ `pare` / `ceda_el_paso` / `limite_velocidad`
4. `camara_vehiculo` $\rightarrow$ `captura` $\rightarrow$ `imagen_capturada`
5. `prediccion_img_X` $\rightarrow$ `asigna_clase` $\rightarrow$ `clase_predicha`

## 3. Conclusión e Interpretación
El modelo de Inteligencia Artificial no solo produce un índice numérico o una clase; ahora cada predicción cuenta con trazabilidad mediante el registro persistente en SQLite y una interpretación semántica válida dentro del dominio vial a través de la ontología.