from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
# import matplotlib.pyplot as plt

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model(
    r"E:\Projects\Potato_Disease_Classification\Training\saved_models\2")
CLASSE_NAMES = ['Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def predict1(img, model, class_names):
    img = tf.image.resize(img, (256, 256))
    img = tf.expand_dims(img, axis=0)
    predict_prob = model.predict(img)
    predict_class = class_names[tf.argmax(predict_prob[0], axis=0).numpy()]
    confidence = np.round(np.max(predict_prob[0]) * 100, 2)

    return predict_class, confidence


# test_img = plt.imread("0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG")
# print(predict(img=test_img, model=MODEL, class_names=CLASSE_NAMES))


@app.post("/predict")
async def predcit(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASSE_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    # file: UploadFile = File(...)
# ):
    # image = read_file_as_image(await file.read())
    # pred, confid = predict1(img=image, model=MODEL, class_names=CLASSE_NAMES)
    # return pred, confid

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
