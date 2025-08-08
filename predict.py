import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('/lung_cancer_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    result = "Cancer" if prediction > 0.5 else "Normal"
    print(f"Prediction: {result} (confidence: {prediction:.2f})")

if __name___ == "__main__":
    predict_image(sys.argv[1])