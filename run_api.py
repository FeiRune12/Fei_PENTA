import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64
# Mantenha esta linha
from geb_1_3b import modeling_geblm 

# Criação da instância da aplicação
app = FastAPI()

# Definição do formato de entrada para o /generate
class GenerationInput(BaseModel):
    prompt: str
    
# URL para a API de Inferência do Stable Diffusion (Exemplo)
# Você pode escolher qualquer modelo de Hugging Face
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1-base"

# Obtém o token da variável de ambiente (Seguro!)
HF_API_KEY = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


# Rota de teste (mantida)
@app.get("/")
def read_root():
    return {"Hello": "World", "Status": "API VIVO!"}


@app.post("/generate")
async def generate_image(data: GenerationInput):
    """
    Gera uma imagem usando a API de Inferência do Hugging Face 
    (O trabalho pesado é feito no servidor deles, não no Render).
    """
    if not HF_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Chave HF_TOKEN não configurada no Render."
        )

    try:
        payload = {"inputs": data.prompt}
        
        # Faz a requisição HTTP POST para a API de Inferência
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        
        # Verifica se a requisição foi bem-sucedida
        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={"status": "error", "message": f"Erro na API Hugging Face: {response.text}"}
            )

        # A resposta é a imagem codificada (em bytes)
        image_bytes = response.content
        
        # Codifica os bytes da imagem para Base64 (para ser facilmente consumido pelo cliente)
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            "status": "success",
            "prompt": data.prompt,
            "image_base64": img_base64,
            "format": "image/jpeg" # ou PNG, depende do modelo
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao processar a requisição: {e}"
        )