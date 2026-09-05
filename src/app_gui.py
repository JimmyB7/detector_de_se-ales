import sys
from pathlib import Path
import tkinter as tk
import customtkinter as ctk

# Configuración de rutas para importar las semanas
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

# Apariencia visual
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class InterfaceSistemaVehicular(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema Inteligente de Control Vehicular y Reconocimiento de Señales")
        self.geometry("900x650")

        # Carga de base de conocimiento (Semana 05)
        try:
            self.docs, self.vectorizer, self.doc_matrix, self.classifier = build_hybrid_system()
        except Exception:
            self.docs = None

        # Contenedor de pestañas
        self.tabview = ctk.CTkTabview(self, width=860, height=600)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)

        # Crear pestañas individuales + pestaña integrada
        self.tab_s2 = self.tabview.add("Semana 02")
        self.tab_s3 = self.tabview.add("Semana 03")
        self.tab_s4 = self.tabview.add("Semana 04")
        self.tab_s5 = self.tabview.add("Semana 05")
        self.tab_full = self.tabview.add("Sistema Completo")

        # Configurar el contenido de cada pestaña
        self.setup_semana02()
        self.setup_semana03()
        self.setup_semana04()
        self.setup_semana05()
        self.setup_sistema_completo()

    # --------------------------------------------------------------------------
    # SEMANA 02: Clasificación de Señales y Reglas
    # --------------------------------------------------------------------------
    def setup_semana02(self):
        lbl = ctk.CTkLabel(self.tab_s2, text="Semana 02: Clasificación Visual y Motor de Reglas", font=("Arial", 16, "bold"))
        lbl.pack(pady=10)

        btn = ctk.CTkButton(self.tab_s2, text="Ejecutar Módulo Semana 02", command=self.evaluar_s2)
        btn.pack(pady=5)

        self.txt_s2 = ctk.CTkTextbox(self.tab_s2, width=780, height=450)
        self.txt_s2.pack(pady=10)

    def evaluar_s2(self):
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)

        out = f"=== EVALUACIÓN DEL MODELO DE CLASIFICACIÓN ===\n"
        out += f"Accuracy: {acc:.3f}\n\n"
        out += f"Matriz de Confusión:\n{cm}\n\n"
        out += f"=== INFERENCIA EN FOTOGRAMAS DE PRUEBA ===\n"
        for i in range(3):
            regla = motor_de_reglas_vehicular(pred[i])
            out += f"Fotograma {i+1} -> Señal: {regla['senial']}\n"
            out += f"  Acción: {regla['accion']}\n"
            out += f"  Prioridad: {regla['prioridad']}\n\n"

        self.txt_s2.delete("1.0", tk.END)
        self.txt_s2.insert(tk.END, out)

    # --------------------------------------------------------------------------
    # SEMANA 03: Taxonomía de IA
    # --------------------------------------------------------------------------
    def setup_semana03(self):
        lbl = ctk.CTkLabel(self.tab_s3, text="Semana 03: Análisis Taxonómico de Casos", font=("Arial", 16, "bold"))
        lbl.pack(pady=10)

        btn = ctk.CTkButton(self.tab_s3, text="Ejecutar Módulo Semana 03", command=self.evaluar_s3)
        btn.pack(pady=5)

        self.txt_s3 = ctk.CTkTextbox(self.tab_s3, width=780, height=450)
        self.txt_s3.pack(pady=10)

    def evaluar_s3(self):
        cases = read_cases()
        out = f"=== ANÁLISIS TAXONÓMICO DE CASOS ===\n\n"
        for i, case in enumerate(cases, start=1):
            primary, detected, _ = classify_problem(case)
            out += f"Caso {i}: {case}\n"
            out += f"  Categoría Principal: {primary}\n"
            out += f"  Áreas Detectadas: {', '.join(detected)}\n\n"

        self.txt_s3.delete("1.0", tk.END)
        self.txt_s3.insert(tk.END, out)

    # --------------------------------------------------------------------------
    # SEMANA 04: Búsqueda A* y Decisiones Minimax
    # --------------------------------------------------------------------------
    def setup_semana04(self):
        lbl = ctk.CTkLabel(self.tab_s4, text="Semana 04: Planificación A* y Decisiones Minimax", font=("Arial", 16, "bold"))
        lbl.pack(pady=10)

        btn = ctk.CTkButton(self.tab_s4, text="Ejecutar Módulo Semana 04", command=self.evaluar_s4)
        btn.pack(pady=5)

        self.txt_s4 = ctk.CTkTextbox(self.tab_s4, width=780, height=450)
        self.txt_s4.pack(pady=10)

    def evaluar_s4(self):
        ruta = planificar_ruta_astar(INICIO, META)
        escenario = ["X", "O", "X", "O", "X", " ", " ", " ", "O"]
        pos = calcular_mejor_maniobra(escenario)

        out = f"=== PLANIFICACIÓN DE RUTA (A*) ===\n"
        out += f"Origen: {INICIO} | Destino: {META}\n"
        out += f"Ruta trazada: {ruta}\n\n"
        out += f"=== EVALUACIÓN DE MANIOBRA (MINIMAX) ===\n"
        out += f"Estado del tráfico: {escenario}\n"
        out += f"Posición óptima seleccionada: {pos}\n"

        self.txt_s4.delete("1.0", tk.END)
        self.txt_s4.insert(tk.END, out)

    # --------------------------------------------------------------------------
    # SEMANA 05: Sistema Híbrido y TF-IDF
    # --------------------------------------------------------------------------
    def setup_semana05(self):
        lbl = ctk.CTkLabel(self.tab_s5, text="Semana 05: Sistema Híbrido (Reglas + TF-IDF)", font=("Arial", 16, "bold"))
        lbl.pack(pady=10)

        self.entry_s5 = ctk.CTkEntry(self.tab_s5, placeholder_text="Ingresa evento o consulta...", width=500)
        self.entry_s5.pack(pady=5)

        btn = ctk.CTkButton(self.tab_s5, text="Procesar Consulta", command=self.evaluar_s5)
        btn.pack(pady=5)

        self.txt_s5 = ctk.CTkTextbox(self.tab_s5, width=780, height=380)
        self.txt_s5.pack(pady=10)

    def evaluar_s5(self):
        if not self.docs:
            self.txt_s5.insert(tk.END, "Error: No se pudo cargar base_conocimiento.txt\n")
            return

        query = self.entry_s5.get()
        if not query:
            query = "Se vislumbra una senal de PARE en la interseccion"

        res = answer_event(query, self.docs, self.vectorizer, self.doc_matrix, self.classifier)

        out = f"Consulta ingresada: \"{query}\"\n\n"
        out += f"- Reglas Disparadas: {res['reglas']}\n"
        out += f"- Evidencia Encontrada: {res['evidencia']}\n"
        out += f"- Similitud TF-IDF: {res['similitud']:.3f}\n"
        out += f"- Categoría Asignada: {res['clase']}\n"

        self.txt_s5.delete("1.0", tk.END)
        self.txt_s5.insert(tk.END, out)

    # --------------------------------------------------------------------------
    # SISTEMA COMPLETO UNIFICADO
    # --------------------------------------------------------------------------
    def setup_sistema_completo(self):
        lbl = ctk.CTkLabel(self.tab_full, text="Ejecución Integral del Sistema Híbrido", font=("Arial", 16, "bold"))
        lbl.pack(pady=10)

        btn = ctk.CTkButton(self.tab_full, text="Ejecutar Flujo Unificado (S2 + S3 + S4 + S5)", command=self.evaluar_sistema_completo)
        btn.pack(pady=5)

        self.txt_full = ctk.CTkTextbox(self.tab_full, width=780, height=450)
        self.txt_full.pack(pady=10)

    def evaluar_sistema_completo(self):
        out = "==========================================================================\n"
        out += "     FLUJO COMPLETO DEL SISTEMA VEHICULAR INTELIGENTE\n"
        out += "==========================================================================\n\n"

        # Semana 02
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        out += f"[SEMANA 02 - CLASIFICACIÓN Y REGLAS]\n"
        out += f"Accuracy del modelo: {acc:.3f}\n"
        regla_s2 = motor_de_reglas_vehicular(pred[0])
        out += f"Ejemplo Inferencia: Señal {regla_s2['senial']} -> {regla_s2['accion']}\n\n"

        # Semana 03
        cases = read_cases()
        out += f"[SEMANA 03 - TAXONOMÍA DE CASOS]\n"
        if cases:
            primary, detected, _ = classify_problem(cases[0])
            out += f"Caso: '{cases[0][:50]}...'\n"
            out += f"Categoría: {primary} | Áreas: {', '.join(detected)}\n\n"

        # Semana 04
        ruta = planificar_ruta_astar(INICIO, META)
        escenario = ["X", "O", "X", "O", "X", " ", " ", " ", "O"]
        pos = calcular_mejor_maniobra(escenario)
        out += f"[SEMANA 04 - A* Y MINIMAX]\n"
        out += f"Ruta trazada (A*): {ruta}\n"
        out += f"Posición de maniobra (Minimax): {pos}\n\n"

        # Semana 05
        out += f"[SEMANA 05 - SISTEMA HÍBRIDO Y BASE DE CONOCIMIENTO]\n"
        if self.docs:
            res_s5 = answer_event("Se vislumbra una senal de PARE en la interseccion", self.docs, self.vectorizer, self.doc_matrix, self.classifier)
            out += f"Reglas activadas: {res_s5['reglas']}\n"
            out += f"Norma / Evidencia: {res_s5['evidencia']}\n"
            out += f"Categoría predicha: {res_s5['clase']}\n\n"

        out += "==========================================================================\n"
        out += "EJECUCIÓN INTEGRADA FINALIZADA CON ÉXITO\n"
        out += "=========================================================================="

        self.txt_full.delete("1.0", tk.END)
        self.txt_full.insert(tk.END, out)


if __name__ == "__main__":
    app = InterfaceSistemaVehicular()
    app.mainloop()