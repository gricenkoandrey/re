from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
from transformers import pipeline

app = FastAPI()

# Инициализация модели генерации изображения
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
