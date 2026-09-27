# Reporte Semana 07: Pipeline Integrado para el Detector de Señales de Tránsito

## 1. Descripción del Problema
El proyecto consiste en el desarrollo de un sistema inteligente enfocado en la detección y clasificación automática de señales de tránsito vial (Pare, Ceda el Paso, Límites de Velocidad, etc.) a partir de imágenes capturadas por cámaras vehiculares.

## 2. Arquitectura del Pipeline y Preprocesamiento
1. **Adquisición y Normalización:** Redimensionamiento de recortes de señales a una resolución uniforme ($32 \times 32$ píxeles en escala de grises).
2. **Extracción de Características:** Vectorización del parche de imagen en un vector de $1024$ características numéricas.
3. **Flujo de Ejecución:** Modularización en `src/` con artefactos persistentes exportados a `artifacts/`.

## 3. Resultados Obtenidos
* Interfaz gráfica unificada desarrollada en `Tkinter` (`app_gui.py`) que permite ejecutar los procesos de prueba e inspeccionar los artefactos de forma centralizada.
* Automatización de la división de datos de entrenamiento ($75\%$) y validación ($25\%$).