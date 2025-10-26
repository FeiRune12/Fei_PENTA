import os
import requests
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# import replicate # Removido para ser gratuito

# --- Configuração da API Craiyon (GRATUITA) ---
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
app = FastAPI(title="FeiPENTA AI Image Generator (Craiyon Free Edition)")

@app.on_event("startup")
async def startup_event():
    """Confirma que a aplicação está inicializada."""
    print("✅ Aplicação FastAPI inicializada. Usando Craiyon API (modelo gratuito).")

@app.get("/")
def read_root():
    return {"Hello": "World", "status": "FeiPENTA Craiyon API Online"}

@app.post("/generate", response_model=ImageResponse)
async def generate_image(request: PromptRequest):
    """
    Gera uma imagem usando a API gratuita do Craiyon.
    Nota: A qualidade é inferior a modelos pagos, mas é 100% gratuita.
    """
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="O prompt é obrigatório.")
        
    try:
        # 1. Preparar o payload para a Craiyon API
        payload = {"prompt": prompt}
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        # 2. Chamar a Craiyon API com um TIMEOUT ALTO (240s = 4 minutos)
        # Isso aumenta a chance de sucesso, já que a API gratuita pode ser lenta.
        print(f"Chamando Craiyon API com prompt: {prompt}")
        
        api_response = requests.post(
            CRAIYON_API_URL,
            headers=headers,
            json=payload,
            timeout=240 # Aumentado para 4 minutos
        )
        api_response.raise_for_status() # Levanta exceção para erros HTTP
        
        result = api_response.json()

        image_list_base64 = result.get('images')
        
        if not image_list_base64 or not image_list_base64[0]:
            error_msg = result.get('error', 'Resposta do Craiyon incompleta ou erro desconhecido.')
            raise HTTPException(status_code=500, detail=f"Erro na Craiyon API: {error_msg}")
            
        # O Base64 retornado pelo Craiyon é uma string pronta para uso
        img_base64 = image_list_base64[0]
            
        # 3. Retornar a resposta no formato ImageResponse
        return {
            "status": "success",
            "prompt": prompt,
            "image_base64": img_base64
        }
        
    except requests.exceptions.Timeout:
        # Erro de Timeout é comum no Craiyon, tratamos explicitamente
        raise HTTPException(status_code=504, detail="A Craiyon API demorou demais para responder. Tente novamente.")
    except requests.exceptions.RequestException as e:
        print(f"ERRO REQUESTS: {e}")
        status_code = e.response.status_code if e.response is not None else 500
        error_detail = e.response.json().get('error', {}).get('message', str(e)) if e.response is not None and e.response.content else str(e)
        
        raise HTTPException(status_code=status_code, detail=f"Erro de comunicação com a Craiyon API ({status_code}). Detalhe: {error_detail}")
    except Exception as e:
        print(f"ERRO INESPERADO: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")
