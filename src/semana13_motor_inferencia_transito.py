from dataclasses import dataclass

@dataclass(frozen=True)
class ReglaTransito:
    nombre: str
    premisas: frozenset[str]
    conclusion: str

def inferir_transito(hechos_iniciales, reglas):
    """
    Motor de inferencia por encadenamiento hacia adelante.
    Recorre iterativamente las reglas hasta que no se puedan derivar nuevos hechos.
    """
    hechos = set(hechos_iniciales)
    trazas = []
    cambios = True

    while cambios:
        cambios = False
        for regla in reglas:
            if regla.premisas.issubset(hechos) and regla.conclusion not in hechos:
                hechos.add(regla.conclusion)
                trazas.append(regla.nombre)
                cambios = True

    return hechos, trazas

# Catálogo de reglas lógicas para el Detector de Señales de Tránsito y Control Vehicular
reglas_senales = [
    ReglaTransito(
        "R1",
        frozenset({"senal_pare_detectada", "linea_detencion_visible"}),
        "requiere_detencion_total"
    ),
    ReglaTransito(
        "R2",
        frozenset({"requiere_detencion_total", "vehiculo_en_movimiento"}),
        "accion_frenar_emergencia"
    ),
    ReglaTransito(
        "R3",
        frozenset({"accion_frenar_emergencia", "via_despejada_tras_3s"}),
        "autorizado_reanudar_marcha"
    )
]

if __name__ == "__main__":
    print("=== PROTOTIPO: MOTOR DE INFERENCIA PARA SEÑALES DE TRÁNSITO ===")

    # Escenario A: Caso completo (Demostrada la detención y el reanudado si se despeja)
    hechos_caso_a = {
        "senal_pare_detectada",
        "linea_detencion_visible",
        "vehiculo_en_movimiento",
        "via_despejada_tras_3s"
    }

    finales, trazas = inferir_transito(hechos_caso_a, reglas_senales)
    print(f"\n[Escenario A] Hechos Iniciales: {hechos_caso_a}")
    print(f"Hechos Derivados Finales: {finales}")
    print(f"Trazas (Reglas Activadas): {trazas}")
    print("-> ESTADO: Demostrado con éxito (Se activó la lógica de frenado y avance posterior).")
