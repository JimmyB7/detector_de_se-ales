# ==========================================
# Archivo: src/semana16_detector_transito_pruebas.py
# Módulo Integral: Detector de Señales y Motor de Referencia
# Proyecto: Detector de Señales de Tránsito
# ==========================================

import re
import unicodedata

# Base de conocimiento normativo (Semana 16: Trazabilidad y Aprobación)
REFERENCIAS_VIALES = [
    {
        "id": "REF-001",
        "titulo": "Regulación de la Señal de PARE (Stop)",
        "claves": ["pare", "detener", "octagonal", "prioridad", "alto", "detencion"],
        "respuesta": "El vehículo debe detenerse completamente antes de la línea de detención y ceder el paso a la vía con prelación.",
        "fuente": "Manual de Normativa Vial y Tránsito - Oficial",
        "validada": True,
    },
    {
        "id": "REF-002",
        "titulo": "Protocolo de Ceda el Paso",
        "claves": ["ceda", "paso", "triangulo", "incorporacion", "ceder"],
        "respuesta": "Reduzca la velocidad y ceda el derecho de vía a los vehículos circundantes antes de incorporarse.",
        "fuente": "Guía de Seguridad Vial Urbana - Oficial",
        "validada": True,
    },
    {
        "id": "REF-003",
        "titulo": "Límite Máximo de Velocidad",
        "claves": ["velocidad", "limite", "maxima", "kmh", "restriccion", "km"],
        "respuesta": "Mantenga la velocidad del vehículo por debajo del límite marcado para evitar sanciones y accidentes.",
        "fuente": "Reglamento de Tránsito y Carreteras - Oficial",
        "validada": True,
    },
    {
        "id": "REF-004",
        "titulo": "Prohibido Girar a la Izquierda",
        "claves": ["girar", "izquierda", "prohibido", "viraje"],
        "respuesta": "Se prohíbe el giro a la izquierda en esta intersección para mantener la fluidez del tráfico.",
        "fuente": "Normativa de Circulación Urbana",
        "validada": True,
    },
    {
        "id": "REF-005",
        "titulo": "Borrador sin validar sobre Estacionamiento",
        "claves": ["estacionar", "parquear", "vehiculo"],
        "respuesta": "Norma interna de parqueo en revisión (Borrador no apto para operar).",
        "fuente": "Borrador interno",
        "validada": False,  # El motor DEBE ignorar este registro estrictamente
    }
]

def normalizar(texto: str) -> set[str]:
    """Normaliza el texto quitando tildes, minúsculas y extrayendo tokens válidos."""
    descompuesto = unicodedata.normalize("NFD", texto.casefold())
    sin_tildes = "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")
    return set(re.findall(r"[a-z0-9]+", sin_tildes))

def motor_de_referencia(consulta: str) -> dict:
    """
    Motor de referencia: Compara palabras con referencias aprobadas (validada=True),
    ordena candidatos y retorna respuesta con trazabilidad de fuente.
    """
    palabras = normalizar(consulta)
    candidatos = []

    for referencia in REFERENCIAS_VIALES:
        # Control de aprobación estricto (Semana 16)
        if not referencia["validada"]:
            continue
        
        claves_norm = set()
        for c in referencia["claves"]:
            claves_norm.update(normalizar(c))

        coincidencias = palabras.intersection(claves_norm)
        if coincidencias:
            candidatos.append((len(coincidencias), referencia["id"], referencia))

    # Si no hay coincidencias o fuentes validadas
    if not candidatos:
        return {
            "estado": "SIN_REFERENCIA",
            "respuesta": "No hay evidencia validada; proceda con precaución estándar.",
            "id": None,
            "fuente": "Sistema de Control IA"
        }

    # Ordenar por mayor número de coincidencias y desempatar por ID
    candidatos.sort(key=lambda x: (-x[0], x[1]))
    _, _, mejor_ref = candidatos[0]

    return {
        "estado": "ENCONTRADA",
        "id": mejor_ref["id"],
        "titulo": mejor_ref["titulo"],
        "respuesta": mejor_ref["respuesta"],
        "fuente": mejor_ref["fuente"]
    }

class DetectorSenalesTransito:
    """Clase principal para la detección y consulta normativa de señales de tránsito."""
    
    def __init__(self):
        print("[INFO] Inicializando Detector de Señales de Tránsito...")

    def clasificar_simulado(self, nombre_senal: str) -> dict:
        """
        Simula la detección visual de la señal y la enlaza automáticamente 
        con la base de conocimiento normativa aprobada.
        """
        print(f"\n[VISION] Procesando detección visual: '{nombre_senal}'")
        
        # Consulta al motor de referencia con trazabilidad estricta
        resultado_norma = motor_de_referencia(nombre_senal)
        
        return {
            "senal_detectada": nombre_senal,
            "estado_normativo": resultado_norma["estado"],
            "id_fuente": resultado_norma.get("id"),
            "titulo": resultado_norma.get("titulo"),
            "accion_vehicular": resultado_norma["respuesta"],
            "fuente_oficial": resultado_norma["fuente"]
        }

if __name__ == "__main__":
    print("=" * 60)
    print(" PRUEBAS: DETECTOR DE SEÑALES + MOTOR DE REFERENCIA ")
    print("=" * 60)
    
    detector = DetectorSenalesTransito()

    # Prueba 1: Señal de Pare
    res1 = detector.clasificar_simulado("Señal de pare octagonal detectada en la vía")
    print(f"Estado: {res1['estado_normativo']} | Fuente: {res1['id_fuente']} - {res1['fuente_oficial']}")
    print(f"Acción vehicular: {res1['accion_vehicular']}")

    # Prueba 2: Ceda el paso
    res2 = detector.clasificar_simulado("Ceda el paso triangular")
    print(f"\nEstado: {res2['estado_normativo']} | Fuente: {res2['id_fuente']} - {res2['fuente_oficial']}")
    print(f"Acción vehicular: {res2['accion_vehicular']}")

    # Prueba 3: Objeto fuera de cobertura / Abstención responsable (Sin inventar respuestas)
    res3 = detector.clasificar_simulado("Anuncio publicitario comercial")
    print(f"\nEstado: {res3['estado_normativo']}")
    print(f"Acción vehicular: {res3['accion_vehicular']}")

    print("\n" + "=" * 60)
    print(" [ÉXITO] Módulo probado y operativo.")
    print("=" * 60)