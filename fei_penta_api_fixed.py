from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import requests

app = FastAPI(
    title="Fei PENTA API",
    description="Backend da IA Fei PENTA - Geração de Imagens com IA",
    version="1.0.0"
)

# -------------------------------------------------------------------
# Permitir requisições do site (CORS)
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ['https://fei-penta.onrender.com'] se quiser limitar
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Endpoint principal /generate
# -------------------------------------------------------------------
@app.post("/generate")
async def generate_image(request: Request):
    """
    Recebe um prompt e retorna uma imagem gerada em base64.
    Exemplo de JSON recebido:
        {"prompt": "um gato samurai bebendo chá"}
    """
    try:
        data = await request.json()
        prompt = data.get("prompt")

        if not prompt:
            return JSONResponse({"error": "Prompt não enviado."}, status_code=400)

        # -------------------------------------------------------------------
        # 🔹 Aqui você chama sua IA geradora (pode ser HuggingFace, OpenAI, etc.)
        # Exemplo de simulação para teste:
        # -------------------------------------------------------------------

        # (Opcional) Exemplo de integração com um endpoint real
        # response = requests.post("https://modelo-de-imagem/api", json={"prompt": prompt})
        # image_base64 = response.json().get("image")

        # Simulação temporária (gera imagem vazia)
        placeholder_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
            "ASsJTYQAAAAASUVORK5CYII="
        )
        image_base64 = f"data:image/png;base64,{placeholder_png}"

        return JSONResponse({"image": image_base64, "prompt": prompt})

    except Exception as e:
        return JSONResponse({"error": f"Ocorreu um erro: {str(e)}"}, status_code=500)
