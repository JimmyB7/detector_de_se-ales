# Reporte Semana 16: Trazabilidad y Aprobación de Referencias

## Objetivo

Completar el sistema con un motor de referencias normativas que relacione las señales detectadas con información vial validada, mostrando la fuente y evitando respuestas sin evidencia aprobada.

## Trabajo realizado

- Se creó `src/semana16_detector_transito_pruebas.py` como módulo integral de detección, consulta normativa y trazabilidad.
- Se definió una base interna de referencias con identificador, título, palabras clave, respuesta, fuente y estado de validación.
- Se implementó `normalizar()` para convertir texto a minúsculas, eliminar tildes y extraer tokens de búsqueda.
- Se implementó `motor_de_referencia()` para:
  - Ignorar estrictamente los registros con `validada = False`.
  - Comparar la consulta con las palabras clave normalizadas.
  - Seleccionar la referencia con mayor cantidad de coincidencias.
  - Devolver el identificador, título, respuesta y fuente del resultado.
  - Abstenerse de inventar una respuesta cuando no existe una referencia aprobada.
- Se añadió `DetectorSenalesTransito.clasificar_simulado()` para simular la detección visual y asociar automáticamente la señal con una referencia normativa.
- Se integró la Semana 16 en el dashboard de Streamlit mediante `pag_s16()`.
- La interfaz permite consultar una descripción vial, muestra la respuesta aprobada y excluye visualmente las referencias no validadas.

## Flujo de consulta

1. El usuario describe una señal o situación vial en la interfaz.
2. `motor_de_referencia()` normaliza la consulta.
3. El motor descarta las referencias que no han sido validadas.
4. Las referencias aprobadas compiten por la mayor cantidad de coincidencias.
5. El sistema devuelve:
   - `ENCONTRADA`, junto con identificador, título, respuesta y fuente, si existe evidencia aprobada.
   - `SIN_REFERENCIA`, si no existe evidencia suficiente.
6. La interfaz presenta el resultado con trazabilidad y estado de abstención.

## Control de aprobaciones

Las fuentes aprobadas incluyen, entre otras, las referencias:

| ID | Referencia | Estado |
|---|---|---|
| `REF-001` | Regulación de la Señal de PARE | Validada |
| `REF-002` | Protocolo de Ceda el Paso | Validada |
| `REF-003` | Límite Máximo de Velocidad | Validada |
| `REF-004` | Prohibido Girar a la Izquierda | Validada |
| `REF-005` | Estacionamiento | No validada |

El registro `REF-005` se utiliza como control negativo: el motor debe ignorarlo por completo y nunca mostrarlo como una fuente aprobada.

## Pruebas

El módulo incluye pruebas ejecutables desde consola:

- Consulta de una señal de pare octagonal.
- Consulta de una señal de ceda el paso.
- Consulta de un objeto sin cobertura normativa.

El resultado esperado es que las dos primeras consultas devuelvan referencias aprobadas con su fuente y que la tercera produzca `SIN_REFERENCIA`.

## Ejecución

Ejecutar las pruebas del módulo:

```bash
python src/semana16_detector_transito_pruebas.py
```

Ejecutar la integración visual:

```bash
streamlit run src/app_gui.py
```

## Archivos relacionados

- `src/semana16_detector_transito_pruebas.py`: motor de referencias, control de aprobación y pruebas.
- `src/app_gui.py`: sección visual de consulta y trazabilidad de la Semana 16.
- `reports/semana16.md`: documentación de objetivos, diseño y pruebas.
