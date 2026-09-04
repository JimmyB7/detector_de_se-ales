from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# Generación del conjunto de datos sintético para clasificación de señales
X, y = make_classification(
    n_samples=1200,
    n_features=10,
    n_classes=3,
    n_informative=8,
    n_redundant=0,
    class_sep=1.8,
    random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# Pipeline de preprocesamiento y modelo de clasificación
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
)

model.fit(X_train, y_train)


def motor_de_reglas_vehicular(clase_predicha: int) -> dict:
    """Asigna una orden de control según el tipo de señal identificada."""
    reglas = {
        0: {
            "senial": "PARE",
            "accion": "Detener vehículo por completo.",
            "prioridad": "CRÍTICA",
        },
        1: {
            "senial": "CEDA EL PASO",
            "accion": "Reducir velocidad y ceder el paso.",
            "prioridad": "ALTA",
        },
        2: {
            "senial": "LÍMITE DE VELOCIDAD",
            "accion": "Ajustar velocidad al límite permitido.",
            "prioridad": "MEDIA",
        },
    }
    return reglas.get(
        clase_predicha,
        {"senial": "DESCONOCIDA", "accion": "Mantener precaución", "prioridad": "BAJA"},
    )


if __name__ == "__main__":
    print("Muestras de entrenamiento:", len(X_train))
    print("Muestras de prueba:", len(X_test))

    pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}\n")

    print("Matriz de confusión:")
    print(confusion_matrix(y_test, pred))
    print()

    print("Simulación de respuesta del sistema:")
    for i in range(3):
        clase = pred[i]
        resultado = motor_de_reglas_vehicular(clase)
        print(f"Fotograma {i+1} -> Señal: {resultado['senial']}")
        print(f"  Acción: {resultado['accion']}")
        print(f"  Prioridad: {resultado['prioridad']}")
        if resultado["senial"] == "PARE":
            print("  [ALERTA]: Ejecutando detención del vehículo")
        print()