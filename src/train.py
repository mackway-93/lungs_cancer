import mlflow
import mlflow.keras
from tensorflow import keras
from tensorflow.keras import layers

mlflow.set_experiment("lung_cancer_detection")

with mlflow.start_run():
    # Simple dummy model just for logging
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(10,)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    accuracy = 0.95  # Example metric

    # Log model and metrics
    mlflow.keras.log_model(model, "lung_cancer_classifier_")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("epochs", 10)

    