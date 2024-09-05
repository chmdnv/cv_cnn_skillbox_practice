import numpy as np
import torch
import cv2

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# load the model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')


# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')) -> dict:
    image = np.array(list(map(int, image[1:-1].split(',')))).reshape((28, 28))
    image = prepare(image)
    cv2.imwrite('output.png', image)
    image = torch.tensor(image).t()
    pred = model.predict(image)
    return {'prediction': pred}


def prepare(image: np.array) -> np.array:
    """Center and crop an image, then dilate and blur"""

    # Image center (horizontal)
    for i in range(len(image) // 2):
        if (image[:, i].sum() != 0) and (image[:, -i].sum() != 0):
            break
        if (image[:, i].sum() != 0) and (image[:, -i].sum() == 0):
            image = np.hstack((np.zeros((image.shape[0], 1)), image[:, :-1]))
        elif (image[:, -i].sum() != 0) and (image[:, i].sum() == 0):
            image = np.hstack((image[:, 1:], np.zeros((image.shape[0], 1))))

    # Image crop
    px = 3
    for i in range(len(image) // 2):
        if sum([image[:px].sum(), image[-px:].sum(), image[:, :px].sum(), image[:, -px:].sum()]) == 0:
            image = image[1:-1, 1:-1]
        else:
            break
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_NEAREST)

    # Dilate and blur
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


# static files
app.mount('/', StaticFiles(directory='/static', html=True), name='static')


# Test run
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
