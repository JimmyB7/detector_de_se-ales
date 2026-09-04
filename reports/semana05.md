# Reporte Semana 05 - Sistema Híbrido de Control

## Evaluaciones Registradas

### Evento 1: "Se vislumbra una senal de PARE en la interseccion"
- **Reglas asociadas:** `detener_vehiculo`
- **Norma / Coincidencia:** Al detectar una senal de PARE o STOP el vehiculo debe detenerse por completo antes de la linea de demarcacion.
- **Similitud TF-IDF:** `0.409`
- **Categoría asignada:** `seguridad_critica`

### Evento 2: "Atencion la camara esta sucia y obstruida por polvo"
- **Reglas asociadas:** `alerta_mantenimiento_sensor`
- **Norma / Coincidencia:** Si la lente de la camara de vision artificial esta obstruida o sucia se debe emitir una alerta de mantenimiento.
- **Similitud TF-IDF:** `0.520`
- **Categoría asignada:** `seguridad_critica`

### Evento 3: "Reducir la marcha en la zona escolar por limite de velocidad"
- **Reglas asociadas:** `reducir_velocidad`
- **Norma / Coincidencia:** La senal de zona escolar obliga a reducir la velocidad a un maximo de treinta kilometros por hora.
- **Similitud TF-IDF:** `0.597`
- **Categoría asignada:** `regulacion_velocidad`

## Observaciones Técnicas
- El uso de TF-IDF permite relacionar términos de entrada con las reglas definidas en la base de conocimiento.
- Las reglas directas garantizan prioridad sobre la salida del modelo estadístico en casos críticos.