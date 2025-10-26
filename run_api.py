import os
import requests
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate

# --- Configuração da API Replicate ---
# O cliente 'replicate' procura automaticamente por 'REPLICATE_API_TOKEN'
# nas variáveis de ambiente. Você deve configurar esta variável no Render.
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# Modelo de imagem que você havia especificado
MODELO_REPLICATE = "black-forest-labs/flux-pro"

# --- Modelos Pydantic ---
class PromptRequest(BaseModel):
    prompt: str

class ImageResponse(BaseModel):
    status: str
    prompt: str
    image_base64: str # A imagem será retornada em formato Base64

# --- Inicialização da API ---
app = FastAPI(title="FeiPENTA AI Image Generator (Replicate Edition)")

# --- Funções de Ajuda ---
def setup_replicate_client():
    """Inicializa o cliente Replicate."""
    if not REPLICATE_API_TOKEN:
        raise RuntimeError(
            "REPLICATE_API_TOKEN não está configurado nas variáveis de ambiente."
        )
    # Não precisamos inicializar explicitamente o cliente se o token estiver na variável de ambiente.
    # A função replicate.run() fará isso implicitamente.
    pass

@app.on_event("startup")
async def startup_event():
    """Verifica se o token está disponível na inicialização."""
    try:
        setup_replicate_client()
    except RuntimeError as e:
        print(f"ERRO DE CONFIGURAÇÃO: {e}")
        # Em um ambiente de produção real, você pode querer desabilitar o endpoint
        # ou forçar o encerramento se a dependência principal (API Key) falhar.

@app.get("/")
def read_root():
    return {"Hello": "World", "status": "FeiPENTA Replicate API Online"}

@app.post("/generate", response_model=ImageResponse)
async def generate_image(prompt_request: PromptRequest):
    """
    Gera uma imagem a partir de um prompt usando o modelo Replicate.
    O resultado da imagem é baixado e retornado em formato Base64.
    """
    prompt = prompt_request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="O prompt é obrigatório.")

    try:
        # 1. Chamar a API da Replicate para gerar a imagem
        print(f"Chamando Replicate com prompt: {prompt}")
        
        # A chamada ao replicate.run() retorna uma lista de URLs de imagens.
        output_urls = replicate.run(
            MODELO_REPLICATE,
            input={
                "prompt": prompt,
                "num_outputs": 1,
                "guidance": 4.5 # Parâmetro opcional comum para modelos de imagem
            }
        )
        
        if not output_urls or not output_urls[0]:
            raise HTTPException(status_code=500, detail="Replicate não retornou um link de imagem válido.")
            
        image_url = output_urls[0]
        
        # 2. Baixar a imagem da URL
        print(f"Baixando imagem da URL: {image_url}")
        image_response = requests.get(image_url, timeout=30)
        image_response.raise_for_status() # Levanta exceção para erros HTTP
        
        # 3. Codificar a imagem em Base64
        img_base64 = base64.b64encode(image_response.content).decode("utf-8")
        
        # 4. Retornar a resposta no formato ImageResponse
        return {
            "status": "success",
            "prompt": prompt,
            "image_base64": img_base64
        }
        
    except replicate.exceptions.ReplicateException as e:
        # Erros específicos do Replicate (ex: token inválido, modelo não encontrado)
        print(f"ERRO REPLICATE: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na API da Replicate. Verifique o token e o modelo. Detalhe: {e}")
    except requests.exceptions.RequestException as e:
        # Erros ao baixar a imagem da URL
        print(f"ERRO REQUESTS: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao baixar a imagem gerada: {e}")
    except Exception as e:
        print(f"ERRO INESPERADO: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")
