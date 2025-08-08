import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Enable autologging
mlflow.tensorflow.autolog()

# Start MLflow run
with mlflow.start_run():

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_data = train_datagen.flow_from_directory(
        'data/train', target_size=(128, 128), class_mode='binary', batch_size=32
    )

    val_data = train_datagen.flow_from_directory(
        'data/val', target_size=(128, 128), class_mode='binary', batch_size=32
    )

    model.fit(train_data, validation_data=val_data, epochs=5)

    