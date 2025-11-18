from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Указываем путь локальной модели
MODEL_PATH = "/home/narana/weights/best_rubert_tiny"

class RubertClassifier:
    def __init__(self):
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )

        # Загружаем модель (она сама найдёт model.safetensors)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )

        self.model.eval()  # переводим в режим инференса

        # Определяем устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text: str):
        # Токенизация
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # Перенос на GPU/CPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

        return {
            "label": int(predicted_class),
            "confidence": float(confidence)
        }
