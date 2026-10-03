from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica para exportar imágenes limpiamente
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, feature, filters, measure

# Configuración de rutas
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def procesar_vision_senal(ruta_imagen, sigma=2.0):
    """
    Procesa una imagen de señal de tránsito realizando:
    1. Conversión a escala de grises.
    2. Detección de bordes mediante Canny.
    3. Umbralización adaptativa mediante el algoritmo de Otsu.
    4. Conteo y etiquetado de regiones conectadas.
    5. Exportación de la evidencia en artifacts/semana09_vision.png.
    """
    path_img = Path(ruta_imagen)
    if not path_img.exists():
        raise FileNotFoundError(f"No se encontró la imagen en: {ruta_imagen}")

    # Cargar imagen y convertir a escala de grises
    pil_img = Image.open(path_img).convert('L')
    image_gray = np.array(pil_img, dtype=np.float32)

    # 1. Detección de bordes con Canny (normalizando intensidad entre 0 y 1)
    image_norm = image_gray / 255.0 if image_gray.max() > 1.0 else image_gray
    edges = feature.canny(image_norm, sigma=sigma)

    # 2. Umbral de Otsu y Máscara Binaria
    threshold = filters.threshold_otsu(image_gray)
    mask = image_gray > threshold

    # 3. Regiones Conectadas
    labels = measure.label(mask)
    num_regiones = int(labels.max())

    # 4. Generación y guardado del gráfico comparativo
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title("Gris / Escala")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title(f"Bordes Canny (σ={sigma})")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Máscara Otsu (T={threshold:.1f})")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    output_path = ARTIFACTS / "semana09_vision.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "umbral_otsu": float(threshold),
        "num_regiones": num_regiones,
        "sigma_canny": sigma,
        "ruta_artefacto": str(output_path)
    }


def ejecutar_practica_semana09():
    """Ejecución de la práctica guiada con dataset estándar (monedas)."""
    image = data.coins()
    edges = feature.canny(image / 255.0, sigma=2.0)
    threshold = filters.threshold_otsu(image)
    mask = image > threshold
    labels = measure.label(mask)

    print("=== SEMANA 09: RECONOCIMIENTO DE IMÁGENES (MÓDULO PRÁCTICA) ===")
    print(f"Umbral Otsu calculado: {threshold:.2f}")
    print(f"Regiones conectadas encontradas: {labels.max()}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny (sigma=2.0)")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Otsu")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    output_path = ARTIFACTS / "semana09_vision.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Evidencia guardada exitosamente en '{output_path}'")


if __name__ == "__main__":
    ejecutar_practica_semana09()