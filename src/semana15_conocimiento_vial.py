import sqlite3
import os

# Ruta de la base de datos persistente exigida en la semana
RUTA_DB = "data/senales_conocimiento.db"

def conectar(ruta=RUTA_DB):
    """Establece la conexión con SQLite asegurando la integridad referencial (Foreign Keys)."""
    directorio = os.path.dirname(os.fspath(ruta))
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    conexion = sqlite3.connect(ruta)
    conexion.execute("PRAGMA foreign_keys = ON;")
    return conexion

def inicializar_base_conocimiento(ruta=RUTA_DB):
    """Crea las tablas relacionales del modelo de conocimiento vial (Semana 15)."""
    with conectar(ruta) as conexion:
        # 1. Tabla Dominio (Áreas temáticas viales)
        conexion.execute('''
            CREATE TABLE IF NOT EXISTS dominio_vial (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL UNIQUE,
                descripcion TEXT NOT NULL
            )
        ''')
        
        # 2. Tabla Aplicación / Sistemas del Vehículo
        conexion.execute('''
            CREATE TABLE IF NOT EXISTS modulo_sistema (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL UNIQUE,
                dominio_id INTEGER NOT NULL REFERENCES dominio_vial(id)
            )
        ''')
        
        # 3. Tabla Expertos (Ingenieros de Tránsito o Validadores)
        conexion.execute('''
            CREATE TABLE IF NOT EXISTS experto_vial (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL,
                correo TEXT NOT NULL
            )
        ''')
        
        # 4. Tabla Artículos (Normativas y protocolos asociados a las señales detectadas)
        conexion.execute('''
            CREATE TABLE IF NOT EXISTS articulo_normativo (
                id INTEGER PRIMARY KEY,
                codigo_senal TEXT NOT NULL,
                titulo TEXT NOT NULL,
                contenido TEXT NOT NULL,
                dominio_id INTEGER NOT NULL REFERENCES dominio_vial(id),
                experto_id INTEGER NOT NULL REFERENCES experto_vial(id),
                fuente TEXT NOT NULL,
                estado TEXT NOT NULL DEFAULT 'BORRADOR' CHECK(estado IN ('BORRADOR', 'VALIDADO'))
            )
        ''')
        conexion.commit()

def cargar_datos_iniciales(ruta=RUTA_DB):
    """Inserta registros iniciales para simular la base de conocimiento."""
    with sqlite3.connect(ruta) as conexion:
        conexion.execute("PRAGMA foreign_keys = ON;")
        
        # Dominios
        conexion.execute("INSERT OR IGNORE INTO dominio_vial (id, nombre, descripcion) VALUES (1, 'Regulación Prioritaria', 'Señales de alto y prelación en intersecciones');")
        conexion.execute("INSERT OR IGNORE INTO dominio_vial (id, nombre, descripcion) VALUES (2, 'Control de Velocidad', 'Límites y restricciones de velocidad en vías');")
        
        # Módulos
        conexion.execute("INSERT OR IGNORE INTO modulo_sistema (id, nombre, dominio_id) VALUES (1, 'Sistema de Frenado Autónomo', 1);")
        conexion.execute("INSERT OR IGNORE INTO modulo_sistema (id, nombre, dominio_id) VALUES (2, 'Control Crucero Adaptativo', 2);")
        
        # Expertos
        conexion.execute("INSERT OR IGNORE INTO experto_vial (id, nombre, correo) VALUES (1, 'Dra. Sofia Ramirez', 's.ramirez@transito-ai.org');")
        
        # Artículos Normativos
        conexion.execute('''
            INSERT OR IGNORE INTO articulo_normativo (id, codigo_senal, titulo, contenido, dominio_id, experto_id, fuente, estado)
            VALUES (1, 'PARE', 'Protocolo de Detención Total', 'El vehículo debe detenerse por completo antes de la línea de pare.', 1, 1, 'Manual de Tránsito Nacional', 'VALIDADO');
        ''')
        conexion.execute('''
            INSERT OR IGNORE INTO articulo_normativo (id, codigo_senal, titulo, contenido, dominio_id, experto_id, fuente, estado)
            VALUES (2, 'LIMITE_VEL', 'Restricción de Velocidad Máxima', 'El vehículo no debe superar el límite establecido en la vía.', 2, 1, 'Código Nacional de Movilidad', 'VALIDADO');
        ''')
        conexion.commit()

def buscar_articulos_validos(dominio_nombre, ruta=RUTA_DB):
    """Realiza la consulta SQL con JOIN tal como exige la presentación, filtrando solo los validados."""
    consulta = '''
        SELECT a.codigo_senal, a.titulo, a.contenido, p.nombre AS responsable, a.fuente, a.estado
        FROM articulo_normativo a
        JOIN dominio_vial d ON d.id = a.dominio_id
        JOIN experto_vial p ON p.id = a.experto_id
        WHERE d.nombre = ? AND a.estado = 'VALIDADO'
    '''
    with conectar(ruta) as conexion:
        cursor = conexion.execute(consulta, (dominio_nombre,))
        return cursor.fetchall()

def validar_articulo_por_id(articulo_id, ruta=RUTA_DB):
    """Permite cambiar un artículo de estado BORRADOR a VALIDADO[cite: 1]."""
    consulta = "UPDATE articulo_normativo SET estado = 'VALIDADO' WHERE id = ?"
    with conectar(ruta) as conexion:
        conexion.execute(consulta, (articulo_id,))
        conexion.commit()

if __name__ == "__main__":
    print("=== INICIALIZANDO BASE DE CONOCIMIENTO VIAL (SEMANA 15) ===")
    inicializar_base_conocimiento()
    cargar_datos_iniciales()
    
    print("\n[Prueba 1] Consultando artículos validados del dominio 'Regulación Prioritaria':")
    resultados = buscar_articulos_validos("Regulación Prioritaria")
    for row in resultados:
        print(f"  -> Señal: {row[0]} | Título: {row[1]} | Responsable: {row[3]} | Estado: {row[5]}")
        
    print("\n¡Base de conocimiento y consultas SQLite ejecutadas con éxito!")