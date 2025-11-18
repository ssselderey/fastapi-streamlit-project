import logging
from contextlib import asynccontextmanager
import io
from fastapi.responses import StreamingResponse
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import cv2

from utils.model_rubert import RubertClassifier
from utils.model_yolo import YOLOFaceDetector


logger = logging.getLogger("uvicorn.info")

rubert_model = None
yolo_model = None


# -----------------------------
# Pydantic Models
# -----------------------------

class TextInput(BaseModel):
    text: str


class TextResponse(BaseModel):
    label: int
    confidence: float


class FaceDetectionResponse(BaseModel):
    detections: list


# -----------------------------
# Lifespan (load models once)
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rubert_model, yolo_model

    rubert_model = RubertClassifier()
    logger.info("RuBERT loaded")

    yolo_model = YOLOFaceDetector()
    logger.info("YOLO model loaded")

    yield

    del rubert_model, yolo_model


app = FastAPI(title="ML API (RuBERT + YOLO)", lifespan=lifespan)


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/clf_text", response_model=TextResponse)
def classify_text(data: TextInput):
    """
    Классификация текста: 0 = негатив, 1 = нейтрал, 2 = позитив
    """
    result = rubert_model.predict(data.text)
    return TextResponse(
        label=result["label"],
        confidence=result["confidence"]
    )


@app.post("/clf_image")
async def detect_face(file: UploadFile = File(...)):
    """
    Детекция лиц с помощью YOLO
    """
    # читаем картинку
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    # YOLO предсказание
    detections = yolo_model.predict(img)

    # если нет лиц — возвращаем оригинал
    if not detections:
        _, buffer = cv2.imencode(".jpg", img_np)
        return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")

    # проходим по каждому лицу
    for det in detections:
        x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])

        # вырезаем фрагмент
        face = img_np[y1:y2, x1:x2]

        # если размер маленький — пропускаем
        if face.size == 0:
            continue

        # размытие (можешь увеличить kernel для сильнее эффекта)
        blurred = cv2.GaussianBlur(face, (51, 51), 30)

        # вставляем обратно
        img_np[y1:y2, x1:x2] = blurred

    # кодируем обратно в jpg
    _, buffer = cv2.imencode(".jpg", img_np)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
