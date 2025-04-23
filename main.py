from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
from transformers import pipeline
from fastapi.responses import StreamingResponse

app = FastAPI()

# Указываем домены, с которых разрешены запросы
origins = [
    "https://rt-dun.vercel.app",  # URL твоего фронтенда
    "https://your-frontend-project.vercel.app",  # Другие домены, если они есть
]

# Добавляем CORS middleware для обработки запросов с других доменов
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Разрешить запросы с этих доменов
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все HTTP методы
    allow_headers=["*"],  # Разрешить все заголовки
)

# Инициализация модели генерации изображений
generator = pipeline("text-to-image", model="CompVis/stable-diffusion-v-1-4-original")

class Prompt(BaseModel):
    prompt: str

@app.post("/api/generate")
async def generate_image(prompt: Prompt):
    # Генерация изображения
    image = generator(prompt.prompt)[0]

    # Преобразуем изображение в байты
    byte_io = BytesIO()
    image.save(byte_io, "PNG")
    byte_io.seek(0)

    return StreamingResponse(byte_io, media_type="image/png")

