class RedSemantica:
    """Red sencilla de hechos y relaciones para la práctica de la semana 14."""

    def __init__(self):
        self.hechos = []

    def agregar_hecho(self, sujeto, relacion, objeto):
        hecho = (sujeto, relacion, objeto)
        if hecho not in self.hechos:
            self.hechos.append(hecho)

    def _superclases(self, sujeto):
        encontradas = set()
        pendientes = [sujeto]
        while pendientes:
            actual = pendientes.pop()
            for origen, relacion, destino in self.hechos:
                if origen == actual and relacion == "es_un" and destino not in encontradas:
                    encontradas.add(destino)
                    pendientes.append(destino)
        return encontradas

    def es_un(self, sujeto, clase):
        return sujeto == clase or clase in self._superclases(sujeto)

    def consultar_propiedad(self, sujeto, propiedad):
        fuentes = [sujeto, *sorted(self._superclases(sujeto))]
        valores = []
        for fuente in fuentes:
            valores.extend(
                objeto
                for origen, relacion, objeto in self.hechos
                if origen == fuente and relacion == propiedad and objeto not in valores
            )
        return valores
