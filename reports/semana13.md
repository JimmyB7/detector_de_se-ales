# Reporte Semana 13: Motor de Inferencia Lógica para Tránsito

## Objetivo

Representar reglas de tránsito mediante hechos y premisas, y deducir acciones de control vehicular aplicando encadenamiento hacia adelante.

## Trabajo realizado

- Se definió `ReglaTransito`, una estructura inmutable con el nombre de la regla, sus premisas y su conclusión.
- Se implementó `inferir_transito`, que parte de un conjunto de hechos, activa repetidamente las reglas cuyas premisas se cumplen y devuelve los hechos derivados junto con la traza de reglas activadas.
- Se creó un catálogo de reglas para el escenario de una señal de PARE: detectar la señal y la línea de detención implica detenerse; si el vehículo está en movimiento, se deriva la acción de frenado de emergencia; si la vía queda despejada después de tres segundos, se autoriza reanudar la marcha.
- Se integró el motor en el dashboard. La interfaz permite seleccionar evidencias, ejecutar la inferencia y consultar los hechos finales y las reglas aplicadas.
- Se agregaron pruebas unitarias para comprobar la derivación de la acción de frenado y el comportamiento cuando falta la evidencia de la línea de detención.

## Archivos relacionados

- `src/semana13_motor_inferencia_transito.py`: reglas y motor de inferencia.
- `src/semana13_test_motor_transito.py`: pruebas unitarias del motor.
- `src/app_gui.py`: interfaz interactiva de la semana 13.

## Validación

Las pruebas unitarias verifican tanto un caso en el que se activa el frenado como otro en el que las premisas incompletas impiden activar reglas.
