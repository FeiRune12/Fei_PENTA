# run_api.py

import os
import requests
import base64  # Necessário para codificar a imagem gerada
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Variáveis de Configuração ---
# O Token deve ser configurado como variável de ambiente (HF_TOKEN) no Render
HF_TOKEN = os.environ.get("HF_TOKEN")

# URL para a API de Inferência do Stable Diffusion 1.4 (comprovada como funcional)
API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# --- Definições de Modelos ---
# Modelo para receber o prompt na requisição POST
class PromptRequest(BaseModel):
    prompt: str

# Modelo para a resposta da API (JSON com a imagem Base64)
class ImageResponse(BaseModel):
    status: str
    prompt: str
    image_base64: str

# --- Inicialização do FastAPI ---
app = FastAPI(title="FeiPENTA AI Image Generator")

# --- Rotas da API ---

@app.get("/")
def read_root():
    return {"Hello": "World", "status": "API V200"}

@app.post("/generate", response_model=ImageResponse)
async def generate_image(prompt_request: PromptRequest):
    """
    Gera uma imagem a partir de um prompt de texto usando a API do Hugging Face.
    """
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Erro de configuração: A variável de ambiente HF_TOKEN não está definida."
        )

    # 1. Requisição para o Hugging Face
    try:
        response = requests.post(
            API_URL, 
            headers=HEADERS, 
            json={"inputs": prompt_request.prompt}
        )
        response.raise_for_status() # Lança exceção para códigos de erro (4xx ou 5xx)
        
    except requests.exceptions.RequestException as req_err:
        # Se o erro for do Hugging Face (404, 403, 500, etc.)
        status_code = response.status_code if 'response' in locals() else 500
        
        # Tentativa de obter detalhes do erro da API
        try:
            error_details = response.json().get("error", "Detalhes do erro não disponíveis.")
        except:
            error_details = "Resposta não JSON."
        
        # Verifica 403 (Permissão) e 404 (Modelo/Endpoint)
        if status_code == 403:
             message = "Erro na autenticação (403): O HF_TOKEN pode não ter permissão 'write'."
        elif status_code == 404:
             message = f"Erro no endpoint (404): O modelo '{API_URL}' não está acessível ou não existe."
        else:
             message = f"Erro HTTP com a API do Hugging Face (Status {status_code})."

        raise HTTPException(
            status_code=status_code,
            detail=f"{message} Detalhes: {error_details}"
        )
    except Exception as e:
        # Erros internos diversos
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao processar a requisição: {e}"
        )

    # 2. Processamento da Imagem (Sucesso)
    
    # O conteúdo da resposta (response.content) são os bytes da imagem (PNG)
    image_bytes = response.content
    
    # Codifica os bytes da imagem para Base64 (para ser facilmente consumido pelo cliente)
    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # 3. Retorna o resultado final
    return {
        "status": "success",
        "prompt": prompt_request.prompt,
        "image_base64": img_base64 
    }