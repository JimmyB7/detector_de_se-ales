# Reporte Semana 10: Reconocimiento de Imágenes y Descriptores de Textura LBP en Señales de Tránsito

## 1. Descripción Técnica
En esta práctica se construyó un pipeline de extracción de características para imágenes de señales de tránsito que unifica tres perspectivas:
* **Regiones y Forma (`measure.label`):** Segmentación por el algoritmo de Otsu y medición del área media/estándar del símbolo central de la señal[cite: 1].
* **Distribución de Intensidad:** Histograma global de 32 rangos normalizados para diferenciar señales claras, oscuras o con alto contraste[cite: 1].
* **Textura LBP (*Local Binary Patterns*):** Descriptor de textura local que caracteriza los bordes y pictogramas interiores de las señales (como las letras de PARE o el triángulo de CEDA EL PASO) independientemente de variaciones globales de luz[cite: 1].

## 2. Artefactos Generados
* `artifacts/semana10_features.npy`: Vector concatenado con la información regional, el histograma de intensidad y el histograma LBP[cite: 1].
* `artifacts/semana10_histograma.png`: Evidencia gráfica de la máscara binaria obtenida y la distribución vectorial[cite: 1].