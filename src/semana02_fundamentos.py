from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Configuración de semilla para reproducibilidad
RANDOM_STATE = 42

# 1. Simulación optimizada de características (mayor separación de clases para subir accuracy)
# Clase 0: PARE (Stop)
# Clase 1: CEDA EL PASO (Yield)
# Clase 2: LÍMITE DE VELOCIDAD (Speed Limit)
X, y = make_classification(
    n_samples=1200,          # Incrementado para mejorar el entrenamiento
    n_features=10,
    n_classes=3,
    n_informative=8,
    n_redundant=0,           # Eliminamos ruido para evitar errores
    class_sep=1.8,           # Aumentamos separación entre las señales
    random_state=RANDOM_STATE,
)

# División de datos en entrenamiento (75%) y prueba (25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# 2. Pipeline de Machine Learning para clasificar la señal visual
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
)

# Entrenamiento del modelo
model.fit(X_train, y_train)


# 3. Motor de Reglas e inferencia simbólica (Acción basada en la IA)
def motor_de_reglas_vehicular(clase_predicha: int) -> dict:
    """Traduce la señal detectada por el modelo de IA en una instrucción de control."""
    diccionario_reglas = {
        0: {
            "senial": "PARE",
            "accion": "Detener vehículo por completo (Frenado de emergencia/total).",
            "prioridad": "CRÍTICA",
        },
        1: {
            "senial": "CEDA EL PASO",
            "accion": (
                "Reducir velocidad y ceder el carril a vehículos prioritarios."
            ),
            "prioridad": "ALTA",
        },
        2: {
            "senial": "LÍMITE DE VELOCIDAD",
            "accion": "Ajustar velocidad crucero a la máxima permitida.",
            "prioridad": "MEDIA",
        },
    }
    return diccionario_reglas.get(
        clase_predicha, {"senial": "DESCONOCIDA", "accion": "Mantener precaución", "prioridad": "BAJA"}
    )


if __name__ == "__main__":
    print("=== PROYECTO: SISTEMA HÍBRIDO DE SEÑALES DE TRÁNSITO ===")
    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de prueba: {len(X_test)}")

    # Predicción de prueba sobre el conjunto de test
    pred = model.predict(X_test)
    print(f"Accuracy del modelo de clasificación: {accuracy_score(y_test, pred):.3f}\n")

    print("--- MATRIZ DE EFECTIVIDAD (CONFUSIÓN) ---")
    print("Filas: Clases Reales | Columnas: Clases Predichas (PARE, CEDA EL PASO, LÍMITE)")
    print(confusion_matrix(y_test, pred))
    print()

    print("--- SIMULACIÓN DEL MOTOR DE INFERENCIA Y ACCIONES ---")
    # Simulamos la respuesta para las primeras 3 detecciones del conjunto de prueba
    for i in range(3):
        clase_detectada = pred[i]
        regla_aplicada = motor_de_reglas_vehicular(clase_detectada)
        print(f"Fotograma de prueba [{i+1}] -> Señal detectada por IA: {regla_aplicada['senial']}")
        print(f"   ↳ Acción ejecutada: {regla_aplicada['accion']}")
        print(f"   ↳ Prioridad de control: {regla_aplicada['prioridad']}")
        if regla_aplicada["senial"] == "PARE":
            print("   🚨 [ALERTA DE SEGURIDAD]: Ejecutando comando HALT_VEHICLE() -> DETENER VEHÍCULO")
        print()