import numpy as np

def ejecutar_prueba_semana07_senales():
    print("=== SEMANA 07: REPRESENTACIONES DEL RECONOCIMIENTO EN SEÑALES DE TRÁNSITO ===\n")

    # 1. REPRESENTACIÓN NUMÉRICA (Características visuales de la señal)
    # Atributos: [Proporción_Rojo (0-1), Proporción_Azul (0-1), FormFactor/Circularidad (0-1)]
    senal_detectada = np.array([0.82, 0.05, 0.95])      # Captura actual
    patron_pare_ideal = np.array([0.85, 0.00, 0.98])    # Patrón ideal de "PARE"
    patron_informativa = np.array([0.05, 0.80, 0.50]) # Patrón ideal de "Información"

    dist_pare = np.linalg.norm(senal_detectada - patron_pare_ideal)
    dist_info = np.linalg.norm(senal_detectada - patron_informativa)

    print("1. REPRESENTACIÓN NUMÉRICA")
    print(f"   - Distancia a 'PARE': {round(float(dist_pare), 3)}")
    print(f"   - Distancia a 'Informativa': {round(float(dist_info), 3)}")
    if dist_pare < dist_info and dist_pare < 0.2:
        print("   -> Clasificación numéricas: Coincide con Señal de PARE")

    # 2. REPRESENTACIÓN SIMBÓLICA (Inferencia lógica y sistema de reglas)
    hechos_detectados = {
        "color_predominante_rojo",
        "forma_octogonal",
        "texto_pare",
        "visibilidad_alta",
        "cerca_de_interseccion"
    }
    
    print("\n2. REPRESENTACIÓN SIMBÓLICA")
    print(f"   - Hechos extraídos: {hechos_detectados}")
    
    # Regla 1: Señal de Alto Total
    if {"color_predominante_rojo", "forma_octogonal"}.issubset(hechos_detectados):
        print("   -> Regla 1 Activada: SEÑAL_PARE_IDENTIFICADA")
        if "cerca_de_interseccion" in hechos_detectados:
            print("   -> Accion Agente: [ALERTA] Detener el vehículo por completo en la línea de parada.")

    # Regla 2: Señal de Zona Escolar
    hechos_zona_escolar = {"color_amarillo", "pictograma_peatones", "velocidad_actual_alta"}
    if {"color_amarillo", "pictograma_peatones"}.issubset(hechos_zona_escolar):
        print("   -> Regla 2 Activada: SEÑAL_ZONA_ESCOLAR_IDENTIFICADA")
        print("   -> Acción Agente: Reducir velocidad a máximo 30 km/h.")

    # 3. RECONOCIMIENTO MEDIANTE AUTÓMATA (FSM)
    # Secuencia válida de señales en tramo controlado:
    # 'P': Preaviso/Zona Escolar -> 'R': Reducción Velocidad -> 'F': Fin de Restricción
    def automata_secuencia_senales(secuencia):
        estado = "q0"  # Estado inicial: Tráfico normal
        transiciones = {
            ("q0", "P"): "q1", # Detecta Preaviso
            ("q0", "R"): "q0",
            ("q0", "F"): "q0",
            
            ("q1", "R"): "q2", # Detecta Reducción tras Preaviso
            ("q1", "P"): "q1",
            ("q1", "F"): "q0",
            
            ("q2", "F"): "q3", # Detecta Fin de Restricción tras Reducción
            ("q2", "P"): "q1",
            ("q2", "R"): "q2",
            
            ("q3", "P"): "q1",
            ("q3", "R"): "q0",
            ("q3", "F"): "q0",
        }
        for simbolo in secuencia:
            estado = transiciones.get((estado, simbolo), "q0")
        
        # El estado q3 representa que la secuencia de zona regulada fue completada correctamente
        return estado == "q3"

    print("\n3. RECONOCIMIENTO MEDIANTE AUTÓMATA")
    secuencias_tramo = ["PRF", "PPRFR", "RFP", "PF"]
    for seq in secuencias_tramo:
        es_valida = automata_secuencia_senales(seq)
        print(f"   - Secuencia de señales '{seq}' completa y coherente: {es_valida}")

if __name__ == "__main__":
    ejecutar_prueba_semana07_senales()