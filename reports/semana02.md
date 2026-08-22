# Documentación de la Práctica - Semana 02: Arquitectura Base, Auditoría y Motor de Reglas

## 1. Resumen del Módulo
Durante la Semana 02, se estableció la arquitectura fundamental del **Detector Inteligente de Señales de Tránsito**. El objetivo principal fue construir los cimientos del proyecto orientados al procesamiento de fotogramas, la clasificación supervisada con machine learning y la ejecución de reglas de control vehicular deterministas para la respuesta operativa en tiempo real.

## 2. Componentes y Estructura Desarrollada
* **Configuración del Entorno y Línea Base (`src/semana02_fundamentos.py`):** 
  * Se estandarizó el uso de **Python 3.13.x** y la estructura modular del repositorio[cite: 1].
  * Se implementó un modelo base de clasificación supervisada utilizando `scikit-learn` para procesar características extraídas de las señales viales[cite: 1].
* **Sistema de Auditoría y Trazabilidad (`artifacts/audit.log`):**
  * Registra de forma estandarizada los eventos críticos del sistema, tales como la captura de fotogramas, la detección de señales y las acciones de control ejecutadas con marcas de tiempo precisas.
* **Motor de Reglas y Control Vehicular:**
  * Desarrollado bajo un enfoque de sistemas expertos basados en reglas lógicas.
  * Traduce la predicción de la IA (por ejemplo, la detección de una señal de Pare o Ceda el Paso) en una instrucción de control operativo (frenado, reducción de velocidad) con su respectiva prioridad de seguridad.

## 3. Integración y Trazabilidad de Eventos
El flujo implementado en esta semana garantiza que cualquier inferencia o procesamiento dentro del sistema de asistencia vial sea transparente y auditable. Los eventos clave registrados incluyen:
* **Captura y Percepción:** Recepción y procesamiento de fotogramas simulados o capturados por la cámara frontal.
* **Inferencia del Modelo:** Ejecución del clasificador para determinar el tipo de señal vial presente en el entorno.
* **Aplicación de Reglas de Control:** Activación automatizada de las instrucciones vehiculares correspondientes.
* **Persistencia de Logs:** Almacenamiento automático en la carpeta `artifacts/` para su posterior revisión técnica.

## 4. Conclusión y Resultados
La arquitectura base construida en la Semana 02 permitió estructurar modularmente el proyecto en el repositorio[cite: 1], facilitando la incorporación posterior de clasificadores taxonómicos y análisis avanzados de machine learning durante la Semana 03.