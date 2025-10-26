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


