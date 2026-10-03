# Reporte Semana 09 - Reconocimiento de Imágenes y Visión Artificial

## 1. Imagen usada
Se procesaron imágenes de prueba del dataset estándar `coins` y capturas reales de señales de tránsito (como señales de PARE y CEDA EL PASO) en escala de grises.

## 2. Umbral Otsu
* **Valor generado:** Calculado automáticamente mediante el análisis bimodal del histograma de intensidades[cite: 1].
* **Interpretación:** Separa con precisión el fondo de la imagen de los elementos de interés (símbolos, letras y bordes de la señal)[cite: 1].

## 3. Regiones
* **Cantidad encontrada:** Obtenida mediante la función `measure.label()` sobre la máscara binaria[cite: 1].
* **Significado:** Muestra los componentes o agrupaciones de píxeles conectados identificados dentro de la señal[cite: 1].

## 4. Sigma (Canny)
* **Efecto de modificar sigma:** Aumentar `sigma` (por ejemplo, a 2.0) incrementa el suavizado gaussiano previo, reduciendo el ruido pero omitiendo bordes finos. Un `sigma` menor retiene detalles de alta frecuencia pero es más susceptible a inconsistencias de textura[cite: 1].

## 5. Limitaciones
Condiciones de iluminación desiguales, sombras o deslumbramientos en la vía pueden alterar el histograma global de Otsu o fragmentar los contornos extraídos por Canny[cite: 1].

## 6. Aplicación en el Proyecto Final
Este pipeline permite aislar las formas geométricas (octágonos, triángulos) y los pictogramas centrales de las señales de tránsito antes de pasarlas por el módulo de clasificación de Inteligencia Artificial[cite: 1, 2, 3].