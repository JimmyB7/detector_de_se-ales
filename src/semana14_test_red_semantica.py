import unittest

from semana14_red_semantica_transito import construir_red_transito


class RedSemanticaTransitoTests(unittest.TestCase):
    def setUp(self):
        self.red = construir_red_transito()

    def test_limit_speed_is_a_regulatory_traffic_sign(self):
        self.assertTrue(self.red.es_un("limite_velocidad", "senal_reglamentaria"))
        self.assertTrue(self.red.es_un("limite_velocidad", "senal_transito"))
        self.assertFalse(self.red.es_un("limite_velocidad", "senal_preventiva"))

    def test_properties_are_inherited_from_parent_classes(self):
        self.assertIn("rojo", self.red.consultar_propiedad("pare", "color_borde"))


if __name__ == "__main__":
    unittest.main()
