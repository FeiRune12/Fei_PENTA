import io
import base64
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# Importação da biblioteca Diffusers
from diffusers import DiffusionPipeline

# Importação do seu módulo local (mantenha, se for usar em outras rotas)
from geb_1_3b import modeling_geblm 

# Criação da instância da aplicação
app = FastAPI()

# Variável global para o pipeline do modelo. Carregamos ele UMA VEZ na inicialização.
image_pipeline = None

# Definição do formato de entrada para o /generate
class GenerationInput(BaseModel):
    prompt: str
    num_inference_steps: int = 10  # Passos curtos para ser rápido
    guidance_scale: float = 7.5

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo de geração de imagem na inicialização do servidor."""
    global image_pipeline
    
    # 🚨 ATENÇÃO: Use um modelo MUITO PEQUENO para testes no Render (que tem memória limitada)
    # Exemplo: um modelo Stable Diffusion Tiny para ver se a infraestrutura funciona.
    MODEL_ID = "hf-internal-testing/tiny-random-stable-diffusion-pipe"
    
    # Você pode tentar um modelo real se tiver um plano pago com mais RAM:
    # MODEL_ID = "runwayml/stable-diffusion-v1-5" 
    
    try:
        image_pipeline = DiffusionPipeline.from_pretrained(MODEL_ID)
        # Tenta usar a GPU se disponível (mas o plano gratuito do Render não tem)
        # image_pipeline.to("cuda") 
        print(f"INFO: Modelo de imagem {MODEL_ID} carregado com sucesso.")
    except Exception as e:
        print(f"ERRO ao carregar o modelo de imagem: {e}")
        # Se o modelo falhar ao carregar, a rota /generate não funcionará, mas a API continuará online.


# Rota de teste (mantida)
@app.get("/")
def read_root():
    return {"Hello": "World", "Status": "API VIVO!"}


@app.post("/generate")
async def generate_image(data: GenerationInput):
    """Gera uma imagem a partir de um prompt e a retorna em Base64."""
    global image_pipeline
    
    if image_pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "O modelo de imagem não está carregado. Verifique os logs de inicialização."}
        )

    try:
        # 1. Gera a imagem
        imagem = image_pipeline(
            data.prompt,
            num_inference_steps=data.num_inference_steps,
            guidance_scale=data.guidance_scale
        ).images[0]
        
        # 2. Converte a imagem para Base64
        # Criamos um buffer em memória (sem salvar no disco)
        buffered = io.BytesIO()
        imagem.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 3. Retorna a imagem codificada
        return {
            "status": "success",
            "prompt": data.prompt,
            "image_base64": img_str,
            "format": "image/png"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Erro durante a geração da imagem: {e}"}
        )