from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, measure
from skimage.feature import local_binary_pattern

# Configuración de rutas dentro del proyecto
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def extraer_vector_caracteristicas_senal(imagen_gris):
   
    # 1. Segmentación por umbral de Otsu para aislar el símbolo/forma de la señal
    threshold = filters.threshold_otsu(imagen_gris)
    mascara_binaria = imagen_gris > threshold

    # 2. Etiquetado de regiones conectadas
    etiquetas = measure.label(mascara_binaria)
    regiones = [r for r in measure.regionprops(etiquetas) if r.area > 30]

    areas = np.array([r.area for r in regiones], dtype=float)
    area_media = float(areas.mean()) if len(areas) else 0.0
    area_std = float(areas.std()) if len(areas) else 0.0
    num_regiones = float(len(regiones))

    # 3. Histograma de intensidades globales (32 rangos normalizados)
    hist_intensidad, _ = np.histogram(
        imagen_gris.ravel(), bins=32, range=(0, 256), density=True
    )

    # 4. Descriptor de Textura LBP (Local Binary Patterns)
    radius = 2
    points = 8 * radius
    lbp_map = local_binary_pattern(
        imagen_gris, points, radius, method="uniform"
    )
    hist_lbp, _ = np.histogram(
        lbp_map.ravel(), bins=np.arange(0, points + 3), density=True
    )

    # 5. Concatenar vector final = Regiones + Intensidad + Textura
    vector_caracteristicas = np.concatenate(
        [[area_media, area_std, num_regiones], hist_intensidad, hist_lbp]
    )

    return vector_caracteristicas, mascara_binaria


def ejecutar_semana10_senales():
    print("=== SEMANA 10: EXTRAER VECTORES DE CARACTERÍSTICAS PARA SEÑALES DE TRÁNSITO ===")

    # Simulación/Carga de una imagen de señal (64x64 px) en escala de grises
    np.random.seed(42)
    imagen_senal = np.random.randint(40, 180, (64, 64), dtype=np.uint8)
    # Forma central simular parche de señal (PARE / CEDA EL PASO)
    imagen_senal[16:48, 16:48] = 230

    # Extraer vector y máscara
    vector, mascara = extraer_vector_caracteristicas_senal(imagen_senal)

    # Guardar matriz .npy en artifacts/
    ruta_npy = ARTIFACTS / "semana10_features.npy"
    np.save(ruta_npy, vector)
    print(f"-> Vector guardado en: {ruta_npy}")
    print(f"-> Dimensión total del vector: {len(vector)} características")
    print(f"   - Métricas de Regiones [0:3]: Área Media = {vector[0]:.2f}, Regiones = {int(vector[2])}")
    print(f"   - Histograma de Intensidad [3:35]: 32 bins")
    print(f"   - Histograma de Textura LBP [35:]: {len(vector)-35} bins")

    # Guardar gráfico de evidencia visual en artifacts/
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(mascara, cmap="gray")
    axes[0].set_title("Máscara Segmentada (Otsu)")
    axes[0].axis("off")

    axes[1].bar(range(len(vector)), vector, color="crimson")
    axes[1].set_title("Vector Final (Región + Intensidad + Textura LBP)")
    axes[1].set_xlabel("Índice de Característica")
    axes[1].set_ylabel("Frecuencia / Valor")

    plt.tight_layout()
    ruta_png = ARTIFACTS / "semana10_histograma.png"
    plt.savefig(ruta_png)
    plt.close()
    print(f"-> Gráfica guardada en: {ruta_png}")


if __name__ == "__main__":
    ejecutar_semana10_senales()