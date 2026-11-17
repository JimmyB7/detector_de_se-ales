# Reporte Semana 15: Base de Conocimiento Vial con SQLite

## Objetivo

Implementar una base de conocimiento persistente que relacione señales de tránsito, dominios normativos, sistemas del vehículo, expertos validadores y artículos técnicos.

## Trabajo realizado

- Se creó `src/semana15_conocimiento_vial.py` como módulo independiente para gestionar el conocimiento vial.
- Se utilizó SQLite para almacenar la información de forma persistente en `data/senales_conocimiento.db`.
- Se habilitaron las claves foráneas mediante `PRAGMA foreign_keys = ON`.
- Se definieron cuatro entidades relacionales:
  - `dominio_vial`: agrupa el conocimiento por áreas temáticas.
  - `modulo_sistema`: representa sistemas del vehículo asociados a un dominio.
  - `experto_vial`: registra responsables o validadores de los artículos.
  - `articulo_normativo`: contiene normas, protocolos y acciones asociadas a una señal.
- Se implementó `inicializar_base_conocimiento()` para crear las tablas si todavía no existen.
- Se añadió `cargar_datos_iniciales()` con dominios de regulación prioritaria y control de velocidad, además de dos artículos normativos validados.
- Se implementó `buscar_articulos_validos()` mediante una consulta SQL con `JOIN` que filtra los registros por dominio y estado `VALIDADO`.
- Se agregó `validar_articulo_por_id()` para cambiar un artículo de `BORRADOR` a `VALIDADO`.

## Modelo de conocimiento

| Tabla | Propósito |
|---|---|
| `dominio_vial` | Clasifica el conocimiento vial por temática. |
| `modulo_sistema` | Relaciona un módulo vehicular con su dominio correspondiente. |
| `experto_vial` | Identifica a la persona responsable de validar la información. |
| `articulo_normativo` | Describe el protocolo asociado a un código de señal y su estado de validación. |

## Ejemplo de datos

- Dominio: `Regulación Prioritaria`
- Señal: `PARE`
- Título: `Protocolo de Detención Total`
- Contenido: el vehículo debe detenerse por completo antes de la línea de pare.
- Estado: `VALIDADO`
- Responsable: `Dra. Sofia Ramirez`

## Validación

Al ejecutar el módulo se inicializan las tablas, se cargan los datos iniciales y se consulta el dominio `Regulación Prioritaria`. La consulta devuelve únicamente artículos cuyo estado sea `VALIDADO`, mostrando el código de señal, el título, el responsable y la fuente asociada.

## Archivos relacionados

- `src/semana15_conocimiento_vial.py`: creación del esquema, carga de datos y consultas SQLite.
- `data/senales_conocimiento.db`: base de conocimiento persistente generada al ejecutar el módulo.
