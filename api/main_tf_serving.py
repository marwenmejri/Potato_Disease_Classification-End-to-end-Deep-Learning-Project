from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests


app = FastAPI()

endpoint = "http://localhost:8505/v1/models/saved_models:predict"

CLASSE_NAMES = ['Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def predict(img, model, class_names):
    img = tf.image.resize(img, (256, 256))
    img = tf.expand_dims(img, axis=0)
    predict_prob = model.predict(img)
    predict_class = class_names[tf.argmax(predict_prob[0], axis=0).numpy()]
    confidence = np.round(np.max(predict_prob[0]) * 100, 2)

    return predict_class, confidence


@app.post("/predict")
async def predcit(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASSE_NAMES[np.argmax(prediction)]
    confidence = np.round((np.max(prediction) * 100), 2)

    return {
        "class": predicted_class,
        "confidence": f"{float(confidence)} %"
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
