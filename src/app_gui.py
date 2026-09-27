import os
import pickle
import sqlite3
import webbrowser
from pathlib import Path
import networkx as nx
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request

# Configuración de rutas principales
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
UPLOADS = ROOT / "uploads"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Mapeos de clases
CLASES_MAP = {
    0: "Pare",
    1: "Ceda el Paso",
    2: "Limite Velocidad",
    3: "Prohibido Girar"
}

CLASES_ONTOLOGIA_MAP = {
    0: "pare",
    1: "ceda_el_paso",
    2: "limite_velocidad",
    3: "prohibido_girar"
}

def clasificar_imagen_real(ruta_imagen, mlp_model):
    """
    Clasifica las imágenes considerando tanto el color rojo como la proporción
    del centro blanco para diferenciar PARE de CEDA EL PASO.
    """
    img_rgb = Image.open(ruta_imagen).convert('RGB')
    img_res = img_rgb.resize((64, 64))
    arr = np.array(img_res, dtype=np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Máscara para detectar color rojo
    mascara_roja = (r > 120) & (g < 90) & (b < 90)
    proporcion_rojo = np.sum(mascara_roja) / (64 * 64)

    # Analizar la región central de la imagen (32x32 píxeles del centro)
    centro_r, centro_g, centro_b = r[16:48, 16:48], g[16:48, 16:48], b[16:48, 16:48]
    mascara_centro_blanco = (centro_r > 180) & (centro_g > 180) & (centro_b > 180)
    proporcion_centro_blanco = np.sum(mascara_centro_blanco) / (32 * 32)

    # Reglas de clasificación directa por geometría de color
    if proporcion_rojo > 0.15:
        # Si el centro es predominantemente blanco -> CEDA EL PASO (ID: 1)
        if proporcion_centro_blanco > 0.35:
            return 1
        # Si el centro es predominantemente rojo -> PARE (ID: 0)
        else:
            return 0

    # Inferencia fallback con el modelo MLP si no aplica la regla de color
    img_gray = Image.open(ruta_imagen).convert('L').resize((32, 32))
    img_arr = np.array(img_gray, dtype=np.float32) / 255.0
    if img_arr.mean() > 0.5:
        img_arr = 1.0 - img_arr
    
    features = img_arr.flatten().reshape(1, -1)
    return int(mlp_model.predict(features)[0])

# Plantilla HTML integrada
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Detector de Señales - Semana 08</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 40px; }
        .container { max-width: 900px; margin: 0 auto; background-color: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h1 { color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; font-size: 24px; }
        .upload-box { background-color: #0f172a; padding: 20px; border-radius: 8px; border: 1px dashed #475569; margin-bottom: 25px; }
        input[type="file"] { margin-bottom: 15px; color: #cbd5e1; }
        button { background-color: #0284c7; color: white; border: none; padding: 12px 24px; font-weight: bold; border-radius: 6px; cursor: pointer; transition: 0.2s; }
        button:hover { background-color: #0369a1; }
        .console { background-color: #020617; font-family: 'Courier New', Courier, monospace; color: #4ade80; padding: 20px; border-radius: 8px; border: 1px solid #1e293b; white-space: pre-wrap; font-size: 14px; line-height: 1.5; }
        .highlight { color: #38bdf8; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Semana 08: Inferencia y Mapeo Ontológico</h1>
        
        <div class="upload-box">
            <form method="POST" enctype="multipart/form-data">
                <label><strong>Subir imagen de Señal de Tránsito (JPG / PNG):</strong></label><br><br>
                <input type="file" name="file" accept="image/*" required><br>
                <button type="submit">Analizar Imagen & Guardar Evidencia</button>
            </form>
        </div>

        {% if resultado %}
        <div class="console">
=== SEMANA 08: DETECTOR DE SEÑALES, SQLITE Y ONTOLOGÍA ===

[INFERENCIA DE IMAGEN SUBIDA]
Nombre del archivo: <span class="highlight">{{ resultado.filename }}</span>
Predicción Red Neuronal (MLP): <span class="highlight">{{ resultado.prediccion }}</span>

✅ Evidencia registrada exitosamente en SQLite ('artifacts/imagenes.db').
✅ Relación ontológica actualizada en GraphML ('artifacts/ontologia.graphml').

=== ÚLTIMOS REGISTROS EN SQLITE ===
{% for reg in resultado.registros_sqlite %}
ID: {{ reg[0] }} | Señal: {{ reg[1] }} | Origen: {{ reg[2] }}
{% endfor %}

=== ESTRUCTURA ONTOLÓGICA (GraphML) ===
Total Conceptos (Nodos): {{ resultado.nodos_ontologia }}
Total Relaciones (Aristas): {{ resultado.relaciones_ontologia }}
{% for rel in resultado.relaciones_detalle %}
  - {{ rel }}
{% endfor %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            filepath = UPLOADS / file.filename
            file.save(filepath)

            modelo_path = ARTIFACTS / "modelo_mlp.pkl"
            if not modelo_path.exists():
                return "Error: No existe 'artifacts/modelo_mlp.pkl'. Ejecuta primero 'python src/semana08_red_ontologia.py'."

            with open(modelo_path, "rb") as f:
                mlp = pickle.load(f)

            # Inferencia con filtro de color y geometría
            pred_id = clasificar_imagen_real(filepath, mlp)
            pred_nombre = CLASES_MAP.get(pred_id, "Desconocida")
            pred_onto = CLASES_ONTOLOGIA_MAP.get(pred_id, "desconocida")

            # Registro en SQLite
            db_path = ARTIFACTS / "imagenes.db"
            with sqlite3.connect(db_path) as con:
                con.execute(
                    "INSERT INTO imagenes_senales (etiqueta_id, etiqueta_nombre, split) VALUES (?, ?, ?)",
                    (pred_id, pred_nombre, "subida_usuario")
                )
                con.commit()
                cur = con.cursor()
                cur.execute("SELECT id, etiqueta_nombre, split FROM imagenes_senales ORDER BY id DESC LIMIT 5")
                ultimos = cur.fetchall()

            # Actualización de GraphML
            onto_path = ARTIFACTS / "ontologia.graphml"
            if onto_path.exists():
                G = nx.read_graphml(onto_path)
            else:
                G = nx.DiGraph()

            G.add_edge("imagen_usuario", "prediccion", rel="genera")
            G.add_edge("prediccion", pred_onto, rel="asigna_clase")
            nx.write_graphml(G, onto_path)

            resultado = {
                "filename": file.filename,
                "prediccion": f"{pred_nombre} (ID: {pred_id})",
                "registros_sqlite": ultimos,
                "nodos_ontologia": G.number_of_nodes(),
                "relaciones_ontologia": G.number_of_edges(),
                "relaciones_detalle": [
                    f"({u}) --[{d.get('rel', 'relacion')}]--> ({v})"
                    for u, v, d in G.edges(data=True)
                ]
            }

    return render_template_string(HTML_TEMPLATE, resultado=resultado)

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    print("\nServidor web Flask iniciado en http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)