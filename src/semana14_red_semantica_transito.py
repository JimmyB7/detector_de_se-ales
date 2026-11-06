from red_semantica import RedSemantica

def construir_red_transito() -> RedSemantica:
    red = RedSemantica()

    # Jerarquías de señales (es_un)
    red.agregar_hecho("pare", "es_un", "senal_reglamentaria")
    red.agregar_hecho("ceda_el_paso", "es_un", "senal_reglamentaria")
    red.agregar_hecho("senal_reglamentaria", "es_un", "senal_transito")

    red.agregar_hecho("limite_velocidad", "es_un", "senal_preventiva")
    red.agregar_hecho("senal_preventiva", "es_un", "senal_transito")

    # Propiedades y herencia (necesita / requiere_accion / color_borde)
    red.agregar_hecho("senal_reglamentaria", "color_borde", "rojo")
    red.agregar_hecho("senal_reglamentaria", "requiere_accion", "detencion_o_ceder")

    red.agregar_hecho("pare", "requiere_accion", "detencion_total_inmediata")
    red.agregar_hecho("limite_velocidad", "color_borde", "rojo_con_negro")
    red.agregar_hecho("limite_velocidad", "requiere_accion", "ajustar_velocidad")

    return red

if __name__ == "__main__":
    print("=== RED SEMÁNTICA APLICADA A SEÑALES DE TRÁNSITO ===")
    red = construir_red_transito()

    print("¿'pare' es una 'senal_transito'?:", red.es_un("pare", "senal_transito"))
    print("Acciones heredadas/propias para 'pare':", red.consultar_propiedad("pare", "requiere_accion"))
    print("Color de borde heredado por 'pare':", red.consultar_propiedad("pare", "color_borde"))
