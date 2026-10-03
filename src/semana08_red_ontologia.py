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
