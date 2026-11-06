import unittest
from semana13_motor_inferencia_transito import ReglaTransito, inferir_transito, reglas_senales

class TestMotorInferenciaTransito(unittest.TestCase):

    def test_caso_frenado_demostrado(self):
        # Evidencias donde la cámara detecta el PARE y el vehículo va en movimiento
        hechos_iniciales = {"senal_pare_detectada", "linea_detencion_visible", "vehiculo_en_movimiento"}
        hechos_finales, trazas = inferir_transito(hechos_iniciales, reglas_senales)

        self.assertIn("accion_frenar_emergencia", hechos_finales)
        self.assertEqual(trazas, ["R1", "R2"])

    def test_caso_sin_linea_no_demostrada_detencion(self):
        # Evidencias incompletas (falta la línea de detención, la regla R1 no se dispara)
        hechos_iniciales = {"senal_pare_detectada", "vehiculo_en_movimiento"}
        hechos_finales, trazas = inferir_transito(hechos_iniciales, reglas_senales)

        self.assertNotIn("accion_frenar_emergencia", hechos_finales)
        self.assertEqual(trazas, [])

if __name__ == "__main__":
    unittest.main()
