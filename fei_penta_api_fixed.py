import os
import requests
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
from replicate.client import Client  # Importação explícita do Cliente

# --- Configuração da API Replicate ---
# O cliente 'replicate' procura a chave em 'REPLICATE_API_TOKEN' nas variáveis de ambiente.
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# Modelo de imagem FLUX Pro
MODELO_REPLICATE = "black-forest-labs/flux-pro"

# Inicialização do cliente Replicate globalmente (será feita na startup)
client = None

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
app = FastAPI(title="Fei PENTA API Image Generator (Replicate Edition)")

@app.on_event("startup")
async def startup_event():
    """Inicializa o cliente Replicate e verifica a disponibilidade do token."""
    global client
    if not REPLICATE_API_TOKEN:
        print("⚠️ AVISO: REPLICATE_API_TOKEN não está configurado. A API Replicate não funcionará.")
        # Se o token não estiver configurado, inicializa com None (a chamada .run falhará, mas o linter pode passar)
    
    # Inicializa o cliente usando o token da variável de ambiente (se existir)
    client = Client(api_token=REPLICATE_API_TOKEN)
    print("✅ Cliente Replicate inicializado.")


@app.get("/")
async def root():
    """Root endpoint para teste rápido."""
    return {"message": "API Fei PENTA rodando para Geração de Imagens (Replicate)!"}

@app.post("/generate", response_model=ImageResponse)
async def generate_image(request: PromptRequest):
    """
    Gera uma imagem usando o Replicate, baixa o resultado e retorna a imagem em Base64.
    """
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="O prompt é obrigatório.")
        
    # Verifica se o cliente foi inicializado (se o token estava presente)
    if not client or not REPLICATE_API_TOKEN:
        raise HTTPException(status_code=503, detail="Serviço Replicate indisponível: Token de API ausente.")

    try:
        # 1. Chamar a API da Replicate usando o objeto cliente
        print(f"Chamando Replicate com prompt: {prompt}")
        
        # O output é uma lista de URLs de imagem
        output_urls = client.run( # MUDANÇA AQUI: usando client.run()
            MODELO_REPLICATE,
            input={
                "prompt": prompt,
                "num_outputs": 1,
                "guidance": 4.5
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
        print(f"ERRO REPLICATE: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na API da Replicate. Detalhe: {e}")
    except requests.exceptions.RequestException as e:
        print(f"ERRO REQUESTS: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao baixar a imagem gerada: {e}")
    except Exception as e:
        print(f"ERRO INESPERADO: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")
