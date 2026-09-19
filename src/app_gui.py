import sys
import webbrowser
from pathlib import Path
from flask import Flask, render_template_string, request

# Configuración de rutas de importación
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from semana02_fundamentos import (
    accuracy_score,
    confusion_matrix,
    model,
    motor_de_reglas_vehicular,
    X_test,
    y_test,
)
from semana03_taxonomia import classify_problem, read_cases
from semana04_marco_tecnologico import (
    INICIO,
    META,
    calcular_mejor_maniobra,
    planificar_ruta_astar,
)
from semana05_marco_tecnologico import answer_event, build_hybrid_system

app = Flask(__name__)

# Carga de la base de conocimiento (Semana 05)
try:
    docs, vectorizer, doc_matrix, classifier = build_hybrid_system()
except Exception:
    docs = None

# Plantilla HTML + CSS + JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema Inteligente Vehicular & Señales</title>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #151c2c;
            --input-bg: #090d16;
            --text-color: #e2e8f0;
            --accent-yellow: #f59e0b;
            --accent-cyan: #38bdf8;
            --border-color: #1e293b;
            --road-line: #334155;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1050px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            padding: 15px 0 25px 0;
            border-bottom: 2px dashed var(--road-line);
            margin-bottom: 25px;
        }
        h1 {
            color: var(--accent-yellow);
            margin: 0 0 8px 0;
            font-size: 1.8rem;
            letter-spacing: 0.5px;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tab-btn {
            padding: 10px 20px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            cursor: pointer;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .tab-btn:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }
        .tab-btn.active {
            background-color: var(--accent-yellow);
            color: #0b0f19;
            border-color: var(--accent-yellow);
        }
        .tab-content {
            display: none;
            background-color: var(--card-bg);
            padding: 25px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }
        .tab-content.active {
            display: block;
        }
        h2 {
            color: var(--accent-cyan);
            margin-top: 0;
            font-size: 1.3rem;
            border-left: 4px solid var(--accent-yellow);
            padding-left: 10px;
        }
        .form-box {
            background-color: rgba(9, 13, 22, 0.6);
            padding: 18px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #cbd5e1;
            font-size: 0.9rem;
        }
        select, input[type="text"] {
            width: 100%;
            padding: 10px;
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            border-radius: 4px;
            box-sizing: border-box;
            margin-bottom: 12px;
            font-size: 0.95rem;
        }
        select:focus, input[type="text"]:focus {
            outline: none;
            border-color: var(--accent-cyan);
        }
        button.submit-btn {
            background-color: var(--accent-cyan);
            color: #0b0f19;
            border: none;
            padding: 10px 18px;
            border-radius: 4px;
            font-weight: bold;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        button.submit-btn:hover {
            opacity: 0.9;
        }
        pre {
            background-color: var(--input-bg);
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            color: #34d399;
            border: 1px solid var(--border-color);
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>SISTEMA INTELIGENTE DE CONTROL VEHICULAR</h1>
            <div class="subtitle">Módulos de Telemetría, Reconocimiento de Señales y Control Autónomo</div>
        </header>
        
        <div class="tabs">
            <button id="btn-s2" class="tab-btn" onclick="openTab(event, 's2')">Semana 02</button>
            <button id="btn-s3" class="tab-btn" onclick="openTab(event, 's3')">Semana 03</button>
            <button id="btn-s4" class="tab-btn" onclick="openTab(event, 's4')">Semana 04</button>
            <button id="btn-s5" class="tab-btn" onclick="openTab(event, 's5')">Semana 05</button>
            <button id="btn-full" class="tab-btn" onclick="openTab(event, 'full')">Resumen Global</button>
        </div>

        <!-- SEMANA 02 -->
        <div id="s2" class="tab-content">
            <h2>Semana 02: Clasificación Visual y Motor de Reglas</h2>
            
            <div class="form-box">
                <form action="/evaluar_s2" method="POST">
                    <label for="clase_id">Simular Selección / Inferencia de Señal:</label>
                    <select name="clase_id" id="clase_id">
                        <option value="0" {% if s2_selected == '0' %}selected{% endif %}>Clase 0: Señal de PARE</option>
                        <option value="1" {% if s2_selected == '1' %}selected{% endif %}>Clase 1: Señal CEDA EL PASO</option>
                        <option value="2" {% if s2_selected == '2' %}selected{% endif %}>Clase 2: LÍMITE DE VELOCIDAD</option>
                    </select>
                    <button type="submit" class="submit-btn">Procesar Fotograma</button>
                </form>
            </div>

            <pre>{{ data_s2 }}</pre>
        </div>

        <!-- SEMANA 03 -->
        <div id="s3" class="tab-content">
            <h2>Semana 03: Análisis Taxonómico de Casos</h2>
            <p>Categorización de casos según la taxonomía funcional de la IA.</p>
            <pre>{{ data_s3 }}</pre>
        </div>

        <!-- SEMANA 04 -->
        <div id="s4" class="tab-content">
            <h2>Semana 04: Planificación A* y Decisiones Minimax</h2>
            <p>Optimización de rutas viales y maniobras en tráfico denso.</p>
            <pre>{{ data_s4 }}</pre>
        </div>

        <!-- SEMANA 05 -->
        <div id="s5" class="tab-content">
            <h2>Semana 05: Sistema Híbrido (Reglas + TF-IDF)</h2>
            
            <div class="form-box">
                <form action="/evaluar_s5" method="POST">
                    <label for="query">Ingrese evento o alerta de tránsito:</label>
                    <input type="text" id="query" name="query" value="{{ s5_query }}" placeholder="Ej: Se vislumbra una senal de PARE en la interseccion...">
                    <button type="submit" class="submit-btn">Consultar Base de Conocimiento</button>
                </form>
            </div>

            <pre>{{ data_s5 }}</pre>
        </div>

        <!-- RESUMEN GLOBAL -->
        <div id="full" class="tab-content">
            <h2>Resumen Integrado de Todas las Semanas</h2>
            <p>Ejecución unificada de la telemetría del sistema vehicular.</p>
            <pre>{{ data_full }}</pre>
        </div>
    </div>

    <script>
        function openTab(evt, tabId) {
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));

            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(btn => btn.classList.remove('active'));

            document.getElementById(tabId).classList.add('active');
            if(evt) {
                evt.currentTarget.classList.add('active');
            } else {
                const btn = document.getElementById('btn-' + tabId);
                if (btn) btn.classList.add('active');
            }
        }

        // Mantener activa la pestaña según la respuesta del servidor (parámetro active_tab)
        const activeTabFromServer = "{{ active_tab }}";
        if (activeTabFromServer) {
            openTab(null, activeTabFromServer);
        } else {
            openTab(null, 's2');
        }
    </script>
</body>
</html>
"""

def obtener_datos_semana02(clase_custom=None):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    
    out = f"=== EVALUACIÓN DEL MODELO DE CLASIFICACIÓN ===\nAccuracy: {acc:.3f}\n\n"
    out += f"Matriz de Confusión:\n{cm}\n\n"
    
    if clase_custom is not None:
        regla = motor_de_reglas_vehicular(int(clase_custom))
        out += "=== INFERENCIA MANUAL SELECCIONADA ===\n"
        out += f"Señal Detectada: {regla['senial']}\n"
        out += f"Acción Asignada: {regla['accion']}\n"
        out += f"Nivel Prioridad: {regla['prioridad']}\n\n"
    
    out += "=== INFERENCIA EN FOTOGRAMAS DE PRUEBA ===\n"
    for i in range(3):
        regla = motor_de_reglas_vehicular(pred[i])
        out += f"Fotograma {i+1} -> Señal: {regla['senial']}\n"
        out += f"  Acción: {regla['accion']}\n"
        out += f"  Prioridad: {regla['prioridad']}\n\n"
    return out

def obtener_datos_semana03():
    cases = read_cases()
    out = "=== ANÁLISIS TAXONÓMICO DE CASOS ===\n\n"
    for i, case in enumerate(cases, start=1):
        primary, detected, _ = classify_problem(case)
        out += f"Caso {i}: {case}\n"
        out += f"  Categoría Principal: {primary}\n"
        out += f"  Áreas Detectadas: {', '.join(detected)}\n\n"
    return out

def obtener_datos_semana04():
    ruta = planificar_ruta_astar(INICIO, META)
    escenario = ["X", "O", "X", "O", "X", " ", " ", " ", "O"]
    pos = calcular_mejor_maniobra(escenario)
    out = f"=== PLANIFICACIÓN DE RUTA (A*) ===\nOrigen: {INICIO} | Destino: {META}\nRuta trazada: {ruta}\n\n"
    out += f"=== EVALUACIÓN DE MANIOBRA (MINIMAX) ===\nEstado del tráfico: {escenario}\nPosición óptima seleccionada: {pos}\n"
    return out

def obtener_datos_semana05(query_custom=None):
    if not docs:
        return "Error: No se pudo cargar base_conocimiento.txt"
    
    query = query_custom if query_custom else "Se vislumbra una senal de PARE"
    res = answer_event(query, docs, vectorizer, doc_matrix, classifier)
    
    out = f'Consulta Procesada: "{query}"\n\n'
    out += f"- Reglas Disparadas: {res['reglas']}\n"
    out += f"- Evidencia Encontrada: {res['evidencia']}\n"
    out += f"- Similitud TF-IDF: {res['similitud']:.3f}\n"
    out += f"- Categoría Asignada: {res['clase']}\n"
    return out

def obtener_datos_resumen_global():
    out = "==========================================================================\n"
    out += "     FLUJO COMPLETO DEL SISTEMA VEHICULAR INTELIGENTE\n"
    out += "==========================================================================\n\n"
    out += f"[SEMANA 02 - CLASIFICACIÓN Y REGLAS]\n{obtener_datos_semana02()}\n"
    out += f"[SEMANA 03 - TAXONOMÍA DE CASOS]\n{obtener_datos_semana03()}\n"
    out += f"[SEMANA 04 - A* Y MINIMAX]\n{obtener_datos_semana04()}\n"
    out += f"[SEMANA 05 - SISTEMA HÍBRIDO]\n{obtener_datos_semana05()}\n"
    out += "==========================================================================\n"
    out += "EJECUCIÓN INTEGRADA FINALIZADA CON ÉXITO\n"
    out += "=========================================================================="
    return out

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        data_s2=obtener_datos_semana02(),
        data_s3=obtener_datos_semana03(),
        data_s4=obtener_datos_semana04(),
        data_s5=obtener_datos_semana05(),
        data_full=obtener_datos_resumen_global(),
        s2_selected="0",
        s5_query="Se vislumbra una senal de PARE",
        active_tab="s2"
    )

@app.route("/evaluar_s2", methods=["POST"])
def evaluar_s2():
    clase_id = request.form.get("clase_id", "0")
    return render_template_string(
        HTML_TEMPLATE,
        data_s2=obtener_datos_semana02(clase_id),
        data_s3=obtener_datos_semana03(),
        data_s4=obtener_datos_semana04(),
        data_s5=obtener_datos_semana05(),
        data_full=obtener_datos_resumen_global(),
        s2_selected=clase_id,
        s5_query="Se vislumbra una senal de PARE en la interseccion",
        active_tab="s2"
    )

@app.route("/evaluar_s5", methods=["POST"])
def evaluar_s5():
    query = request.form.get("query", "")
    return render_template_string(
        HTML_TEMPLATE,
        data_s2=obtener_datos_semana02(),
        data_s3=obtener_datos_semana03(),
        data_s4=obtener_datos_semana04(),
        data_s5=obtener_datos_semana05(query),
        data_full=obtener_datos_resumen_global(),
        s2_selected="0",
        s5_query=query,
        active_tab="s5"
    )

if __name__ == "__main__":
    port = 5000
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(port=port, debug=False)