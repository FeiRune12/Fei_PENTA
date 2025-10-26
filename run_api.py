import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# Importação da biblioteca Diffusers
from diffusers import DiffusionPipeline

# Importação do seu módulo local (se houver outras rotas)
from geb_1_3b import modeling_geblm 

# Criação da instância da aplicação
app = FastAPI()

# Variável global para o pipeline do modelo. Inicializada como None.
image_pipeline = None
MODEL_ID = "hf-internal-testing/tiny-random-stable-diffusion-pipe"


# Definição do formato de entrada para o /generate
class GenerationInput(BaseModel):
    prompt: str
    num_inference_steps: int = 10
    guidance_scale: float = 7.5


# Rota de teste
@app.get("/")
def read_root():
    return {"Hello": "World", "Status": "API VIVO!"}


@app.post("/generate")
async def generate_image(data: GenerationInput):
    """Gera uma imagem a partir de um prompt, carregando o modelo sob demanda."""
    global image_pipeline
    
    # 1. Carregamento do modelo (Lazy Loading e Cache)
    if image_pipeline is None:
        try:
            print(f"INFO: Carregando modelo {MODEL_ID} no primeiro acesso...")
            # Esta linha fará o download/cache do modelo
            image_pipeline = DiffusionPipeline.from_pretrained(MODEL_ID)
            print("INFO: Modelo de imagem carregado com sucesso.")
        except Exception as e:
            # Em caso de falha, retorne 500 ou 503
            print(f"ERRO FATAL ao carregar o modelo de imagem: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Falha ao carregar o modelo de imagem. Verifique os logs. Erro: {e}"
            )

    try:
        # 2. Gera a imagem
        imagem = image_pipeline(
            data.prompt,
            num_inference_steps=data.num_inference_steps,
            guidance_scale=data.guidance_scale
        ).images[0]
        
        # 3. Converte a imagem para Base64
        buffered = io.BytesIO()
        imagem.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 4. Retorna a imagem codificada
        return {
            "status": "success",
            "prompt": data.prompt,
            "image_base64": img_str,
            "format": "image/png"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante a geração da imagem: {e}"
        )