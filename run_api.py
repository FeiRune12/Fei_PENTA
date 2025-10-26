from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

# Inicializa o modelo em CPU
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # força CPU

app = FastAPI()

# Permitir que o site acesse a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou coloque seu site
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(request: PromptRequest):
    prompt = request.prompt

    # Gera imagem real em CPU
    image = pipe(prompt, num_inference_steps=20, height=512, width=512).images[0]

    # Salva em memória
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "prompt": prompt,
        "image": f"data:image/png;base64,{img_str}"
    }
