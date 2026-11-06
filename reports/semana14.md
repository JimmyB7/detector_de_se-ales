# Reporte Semana 14: Red Semántica de Señales de Tránsito

## Objetivo

Representar el conocimiento sobre señales de tránsito mediante relaciones semánticas, consultar jerarquías y mostrar las propiedades asociadas a cada señal.

## Trabajo realizado

- Se definieron relaciones `es_un` para organizar las señales en una jerarquía: PARE y CEDA EL PASO como señales reglamentarias, y límite de velocidad como señal preventiva; ambas categorías pertenecen al concepto general de señal de tránsito.
- Se asociaron propiedades a las categorías y señales, como el color del borde y la acción requerida. Por ejemplo, PARE tiene una acción propia de detención total inmediata, mientras que también puede heredar propiedades de su categoría reglamentaria.
- Se implementó en el dashboard una vista interactiva para seleccionar una señal y consultar sus tipos superiores y las propiedades directas o heredadas.
- Se construyó una visualización en forma de grafo dirigido. Las flechas muestran las relaciones y la vista destaca la señal seleccionada y su jerarquía.
- Se añadió `construir_red_transito()` para registrar los hechos usando `RedSemantica`, junto con un ejemplo de consulta en el módulo de la semana. La vista del dashboard construye su grafo por separado con NetworkX.

## Archivos relacionados

- `src/semana14_red_semantica_transito.py`: hechos y función de construcción de la red semántica.
- `src/app_gui.py`: consulta interactiva, visualización del grafo y presentación de propiedades.

## Resultado

La semana amplía el sistema con una representación explícita de categorías y atributos, haciendo posible explorar relaciones entre señales y distinguir propiedades propias de las heredadas.
