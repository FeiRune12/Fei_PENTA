import os
import requests
import base64
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Removemos a necessidade de 'replicate' e 'GEMINI_API_KEY'

# --- Configuração da API Craiyon ---
# O Craiyon API é gratuito e não requer chave de API.
CRAIYON_API_URL = "https://backend.craiyon.com/generate" 

# --- Modelos Pydantic ---

class PromptRequest(BaseModel):
    """Modelo de entrada para o prompt de geração de imagem."""
    prompt: str

class ImageResponse(BaseModel):
    """Modelo de saída para o Base64 da imagem gerada."""
    status: str
    prompt: str
    image_base64: str

# --- Inicialização da API ---
app = FastAPI(title="Fei PENTA API Image Generator (Craiyon Free Edition)")

@app.on_event("startup")
async def startup_event():
    """Confirma que a aplicação está inicializada."""
    print("✅ Aplicação FastAPI inicializada. Usando Craiyon API (modelo gratuito).")

@app.get("/")
async def root():
    """Root endpoint para teste rápido."""
    return {"message": "API Fei PENTA rodando para Geração de Imagens (Craiyon)!"}

@app.post("/generate", response_model=ImageResponse)
async def generate_image(request: PromptRequest):
    """
    Gera uma imagem usando a API gratuita do Craiyon.
    Nota: A qualidade é inferior a SDXL/Gemini, mas é 100% gratuita.
    """
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="O prompt é obrigatório.")
        
    try:
        # 1. Preparar o payload para a Craiyon API
        # Craiyon requer que o payload seja um dicionário simples com o prompt
        payload = {"prompt": prompt}
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        # 2. Chamar a Craiyon API
        print(f"Chamando Craiyon API com prompt: {prompt}")
        
        # O Craiyon é lento, então o timeout é de 120 segundos
        api_response = requests.post(
            CRAIYON_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        api_response.raise_for_status() # Levanta exceção para erros HTTP
        
        result = api_response.json()

        # O Craiyon retorna uma lista de 9 imagens em base64. Pegamos a primeira.
        # A chave é 'images' e contém uma lista de strings Base64 puras.
        image_list_base64 = result.get('images')
        
        if not image_list_base64 or not image_list_base64[0]:
            error_msg = result.get('error', 'Resposta do Craiyon incompleta ou erro desconhecido.')
            raise HTTPException(status_code=500, detail=f"Erro na Craiyon API: {error_msg}")
            
        # 3. O Base64 retornado pelo Craiyon não precisa de re-codificação
        img_base64 = image_list_base64[0]
            
        # 4. Retornar a resposta no formato ImageResponse
        return {
            "status": "success",
            "prompt": prompt,
            "image_base64": img_base64
        }
        
    except requests.exceptions.RequestException as e:
        print(f"ERRO REQUESTS: {e}")
        status_code = e.response.status_code if e.response is not None else 500
        error_detail = e.response.json().get('error', {}).get('message', str(e)) if e.response is not None and e.response.content else str(e)
        
        raise HTTPException(status_code=status_code, detail=f"Erro de comunicação com a Craiyon API ({status_code}). Detalhe: {error_detail}")
    except Exception as e:
        print(f"ERRO INESPERADO: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")
