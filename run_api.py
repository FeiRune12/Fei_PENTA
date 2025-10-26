import os
import requests
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Configuração ---
HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = "https://huggingface.co/spaces/Qwen/Qwen-Image"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# --- Modelos ---
class PromptRequest(BaseModel):
    prompt: str

class ImageResponse(BaseModel):
    status: str
    prompt: str
    image_base64: str

# --- Inicialização da API ---
app = FastAPI(title="FeiPENTA AI Image Generator (Qwen Edition)")

@app.get("/")
def read_root():
    return {"Hello": "World", "status": "FeiPENTA Qwen API Online"}

@app.post("/generate", response_model=ImageResponse)
async def generate_image(prompt_request: PromptRequest):
    """
    Gera uma imagem a partir de um prompt usando o Space Qwen/Qwen-Image.
    """
    if not prompt_request.prompt:
        raise HTTPException(status_code=400, detail="O prompt é obrigatório.")

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": prompt_request.prompt},
            timeout=120
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erro ao conectar ao Space Qwen: {e}")

    # Verifica se o retorno é uma imagem ou JSON de erro
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("image/"):
        # Codifica a imagem em base64
        img_base64 = base64.b64encode(response.content).decode("utf-8")
        return {
            "status": "success",
            "prompt": prompt_request.prompt,
            "image_base64": img_base64
        }
    else:
        try:
            erro = response.json().get("error", "Resposta inesperada do modelo.")
        except Exception:
            erro = "O Space retornou uma resposta não reconhecida."
        raise HTTPException(status_code=500, detail=f"Falha ao gerar imagem: {erro}")
