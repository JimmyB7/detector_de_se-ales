# Código fuente: Semanas 07 a 11

Este documento reúne el contenido completo de los módulos Python de las semanas 07, 08, 09, 10 y 11, en el orden del proyecto. El código corresponde a los archivos de `src/` y se conserva separado por semana para facilitar su lectura y copia.

## Semana 07 — `src/semana07.py`

```python
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
```

## Semana 08 — `src/semana08_red_ontologia.py`

```python
import pickle
import sqlite3
import networkx as nx
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def generar_dataset_robusto(n_samples=1600):
    """
    Genera un dataset de vectores de características sintéticos (Rojo, Forma, Bordes)
    Clases:
    0: Pare (Alto color rojo + bordes densos)
    1: Ceda el Paso (Color blanco/rojo en triángulo)
    2: Limite Velocidad (Círculo rojo/blanco con dígitos)
    3: Prohibido Girar (Banda diagonal)
    """
    np.random.seed(42)
    X, y = [], []
    
    for i in range(n_samples):
        label = i % 4
        base = np.zeros((32, 32))
        
        if label == 0:  # PARE: Octágono sólido relleno
            base[2:30, 2:30] = 0.95
            base[12:20, 10:22] = 0.05
        elif label == 1:  # CEDA EL PASO: Triángulo
            for r in range(32):
                base[r, :min(r + 1, 32)] = 0.8
        elif label == 2:  # LIMITE VELOCIDAD: Anillo circular
            y_grid, x_grid = np.ogrid[:32, :32]
            mask = (x_grid - 16)**2 + (y_grid - 16)**2 <= 13**2
            base[mask] = 0.9
            mask_in = (x_grid - 16)**2 + (y_grid - 16)**2 <= 8**2
            base[mask_in] = 0.1
        elif label == 3:  # PROHIBIDO GIRAR: Cruz
            np.fill_diagonal(base, 0.9)
            np.fill_diagonal(np.fliplr(base), 0.9)

        noise = np.random.normal(0, 0.02, (32, 32))
        img = np.clip(base + noise, 0, 1)
        X.append(img.flatten())
        y.append(label)
        
    return np.array(X), np.array(y)

def ejecutar_semana08_senales():
    print("=== SEMANA 08: REENTRENANDO MODELO DE SEÑALES ===")
    
    X, y = generar_dataset_robusto()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"\n[1/3] Modelo MLP Entrenado. Accuracy: {acc:.2f}")

    with open(ARTIFACTS / "modelo_mlp.pkl", "wb") as f:
        pickle.dump(mlp, f)
    print("      -> Guardado en 'artifacts/modelo_mlp.pkl'")

    db_path = ARTIFACTS / "imagenes.db"
    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS imagenes_senales ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, etiqueta_id INTEGER, etiqueta_nombre TEXT, split TEXT)"
        )
        con.commit()
    print("[2/3] Base de datos verificada.")

    G = nx.DiGraph()
    G.add_edges_from([
        ("senal_transito", "reglamentaria", {"rel": "es_tipo"}),
        ("reglamentaria", "pare", {"rel": "contiene_clase"}),
        ("reglamentaria", "ceda_el_paso", {"rel": "contiene_clase"}),
        ("reglamentaria", "limite_velocidad", {"rel": "contiene_clase"}),
        ("reglamentaria", "prohibido_girar", {"rel": "contiene_clase"}),
        ("detector_mlp", "senal_transito", {"rel": "clasifica"})
    ])
    nx.write_graphml(G, ARTIFACTS / "ontologia.graphml")
    print(f"[3/3] Ontología exportada con éxito.\n")

if __name__ == "__main__":
    ejecutar_semana08_senales()
```

## Semana 09 — `src/semana09_vision.py`

```python
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
```

## Semana 10 — `src/semana10_reconocimiento.py`

```python
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
```

## Semana 11 — `src/semana11_benchmark.py`

```python
import os
from pathlib import Path
from time import perf_counter
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Asegurar directorios de evidencias
Path("artifacts").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# Mapas de etiquetas de señales
CLASSES_MAP = {
    0: "0_pare",
    1: "1_ceda_el_paso",
    2: "2_limite_velocidad",
    3: "3_prohibido_girar"
}

def cargar_dataset_senales(base_path="data/img"):
    """
    Carga y vectoriza las imágenes de señales de tránsito desde data/img/.
    Si no encuentra imágenes suficientes, genera muestras sintéticas para el benchmark.
    """
    X_data = []
    y_data = []
    
    if os.path.exists(base_path):
        for class_id, class_folder in CLASSES_MAP.items():
            folder_path = os.path.join(base_path, class_folder)
            if os.path.exists(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, file_name)
                        try:
                            img = Image.open(img_path).convert('L').resize((32, 32))
                            vector = np.array(img, dtype=np.float32).flatten() / 255.0
                            X_data.append(vector)
                            y_data.append(class_id)
                        except Exception:
                            pass

    # Si hay pocas o ninguna imagen, se simulan datos para garantizar la ejecución del benchmark
    if len(X_data) < 20:
        print("[!] Generando muestras de respaldo para completar la evaluación de benchmarking...")
        np.random.seed(42)
        X_data = np.random.rand(120, 1024)
        y_data = np.random.randint(0, 4, size=120)

    return np.array(X_data), np.array(y_data)

# 2. Cargar datos de señales y realizar split controlado
X, y = cargar_dataset_senales()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Modelos comparados bajo las mismas reglas
models = {
    "logistic": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1500, random_state=42)
    ),
    "forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "mlp": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
    ),
}

# 4. Evaluación de rendimiento y tiempos de entrenamiento
rows = []
for name, model in models.items():
    start = perf_counter()
    model.fit(X_train, y_train)
    elapsed = perf_counter() - start
    acc = accuracy_score(y_test, model.predict(X_test))
    rows.append({
        "modelo": name,
        "accuracy": round(acc, 4),
        "fit_seconds": round(elapsed, 4)
    })

# 5. Exportar evidencia a CSV
df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
print("\n--- RESULTADOS DEL BENCHMARK DE SEÑALES DE TRÁNSITO ---")
print(df.to_string(index=False))
df.to_csv("artifacts/benchmark.csv", index=False)

# 6. Generación del reporte ADR de decisión técnica
adr_content = f"""# ADR Semana 11 - Selección de Modelo para Detección de Señales de Tránsito

## 1. Contexto
Se evalúan tres alternativas de aprendizaje automático para el módulo de clasificación visual del vehículo autónomo, considerando exactitud predictiva y latencia/tiempo de entrenamiento.

## 2. Opciones Evaluadas
- **Logistic Regression**: Clasificador lineal de baja complejidad y alta velocidad.
- **Random Forest**: Ensamble de árboles con alta tolerancia al ruido en imágenes.
- **MLPClassifier**: Red neuronal multicapa para patrones no lineales en vectores de píxeles.

## 3. Evidencia Experimental
Tabla extraída de `artifacts/benchmark.csv`:

| Modelo | Accuracy | Tiempo de Entrenamiento (s) |
| :--- | :--- | :--- |
| {df.iloc[0]['modelo']} | {df.iloc[0]['accuracy']} | {df.iloc[0]['fit_seconds']} |
| {df.iloc[1]['modelo']} | {df.iloc[1]['accuracy']} | {df.iloc[1]['fit_seconds']} |
| {df.iloc[2]['modelo']} | {df.iloc[2]['accuracy']} | {df.iloc[2]['fit_seconds']} |

## 4. Decisión Técnica
Se selecciona **{df.iloc[0]['modelo']}** como el algoritmo principal del sistema por ofrecer el mejor equilibrio entre precisión de detección y costo computacional.

## 5. Consecuencias
- **Ventajas**: Desempeño verificado sobre el mismo conjunto de datos y partición de pruebas.
- **Restricciones**: Monitorear la latencia de inferencia durante la integración en tiempo real dentro del controlador del vehículo.
"""

with open("reports/semana11_adr.md", "w", encoding="utf-8") as f:
    f.write(adr_content)

print("\n[✓] Artefactos generados exitosamente: artifacts/benchmark.csv y reports/semana11_adr.md")
```
