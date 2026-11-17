# Detector Inteligente de Señales de Tránsito

Proyecto académico de demostración para explorar clasificación de señales, visión artificial, reglas de tránsito, bases de conocimiento y trazabilidad. El dashboard principal está construido con Streamlit.

## Requisitos

- Python 3.10 o superior
- Dependencias de `requirements.txt`

## Instalación y ejecución

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run src/app_gui.py
```

También se pueden ejecutar las prácticas individuales, por ejemplo:

```powershell
python src\semana08_red_ontologia.py
python src\semana14_red_semantica_transito.py
```

## Dashboard

El panel lateral permite explorar las semanas implementadas: 02-05, 07-11 y 13-16. El módulo **Sistema Integrado** combina la clasificación, las acciones asociadas a las señales, el registro SQLite, la ontología y las referencias normativas disponibles.

La clasificación automática requiere `artifacts/modelo_mlp.pkl` con un estimador de cuatro clases (etiquetas `0` a `3`) entrenado con imágenes de `32 × 32` píxeles en escala de grises. Si el archivo falta o no coincide con ese contrato, el dashboard se abstiene de predecir y explica el problema. Para generar el modelo demostrativo incluido en la práctica, ejecuta `python src\semana08_red_ontologia.py`.

El modelo de la semana 08 se entrena con datos sintéticos; sus resultados son demostrativos y no deben utilizarse para controlar un vehículo real.

## Estructura

- `src/`: dashboard y módulos de las prácticas.
- `data/`: bases de conocimiento y datos de entrada.
- `artifacts/`: modelos, registros y otros resultados generados.
- `reports/`: reportes y decisiones técnicas.

## Autores

- Jimmy Alexander Beltran Beltran
- Ivan Andres Salas Leiva
- ETITC · 10.º semestre
