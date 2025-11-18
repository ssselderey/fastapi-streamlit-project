import streamlit as st
import requests
from PIL import Image
import io



BACKEND_URL = "http://158.160.154.174:8000"

st.title("FastAPI + Streamlit")
st.write("Классификация текста и изображений")



st.header("Классификация отзывов")

LABELS = {
    0: ("Негативный", "#ff4b4b"),
    1: ("Нейтральный", "#888888"),
    2: ("Позитивный", "#4caf50")
}

user_text = st.text_area("Введите отзыв:")

if st.button("Классифицировать отзыв"):
    if user_text.strip():
        response = requests.post(
            f"{BACKEND_URL}/clf_text",
            json={"text": user_text}
        )

        if response.status_code == 200:
            result = response.json()

            label_id = result["label"]
            confidence = result.get("confidence", None)

            label_name, color = LABELS.get(label_id, ("Неизвестно", "#ffffff"))

            # Красивый цветной блок
            st.markdown(
                f"""
                <div style="
                    padding: 15px;
                    border-radius: 10px;
                    background-color: {color}22;
                    border-left: 8px solid {color};
                    font-size: 22px;
                    font-weight: bold;
                    margin-top: 15px;
                ">
                    Результат: {label_name}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Вероятность
            if confidence is not None:
                st.markdown(f"#### Доверие модели: **{confidence*100:.2f}%**")

            # Полный JSON
            st.json(result)

        else:
            st.error("Ошибка при запросе к серверу")
    else:
        st.warning("Введите текст!")





st.title("Размытие лица YOLO")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Исходное изображение", use_column_width=True)

    if st.button("Обработать"):
        files = {"file": uploaded_file.getvalue()}

        response = requests.post("http://158.160.154.174:8000/clf_image", files=files)

        if response.status_code != 200:
            st.error("Ошибка backend")

        else:
            img = Image.open(io.BytesIO(response.content))
            st.image(img, caption="Размыленное изображение", use_column_width=True)
