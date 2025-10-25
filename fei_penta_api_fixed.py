from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from geb_1_3b.modeling_geblm import GEBLMForCausalLM
import datetime
import os

# --- Configuração do FastAPI ---
app = FastAPI(title="Fei PENTA API Fake Realista")

# --- Modelo de request ---
class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0

# --- Inicialização do modelo fake ---
MODEL_NAME = "GEB-AGI/geb-1.3b"
device = "cpu"

print("🔹 Carregando modelo e tokenizer...")
model = GEBLMForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.to(device)
print("✅ Modelo carregado (fake)")

# --- Diretório e arquivo de log ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "acquisitions.log")

def log_acquisition(prompt: str, response: str):
    """Registra cada prompt e resposta com timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] PROMPT: {prompt} | RESPONSE: {response}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(log_line, end="")  # Também exibe no terminal

# --- Função de geração de resposta fake mais realista ---
def generate_response(prompt: str, max_length: int = 100, temperature: float = 1.0):
    prompt_lower = prompt.lower()
    if "olá" in prompt_lower:
        response = "Olá! Como posso ajudar você hoje?"
    elif "ajuda" in prompt_lower:
        response = "Claro! Estou aqui para te auxiliar."
    elif "teste" in prompt_lower:
        response = "Este é um teste do modelo fake. Tudo funcionando!"
    else:
        response = f"[RESPOSTA SIMULADA] Prompt: {prompt}, max_length: {max_length}, temperature: {temperature}"
    return response

# --- Endpoint principal ---
@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt não pode ser vazio")

    response = generate_response(prompt, request.max_length, request.temperature)

    # Log de aquisição
    log_acquisition(prompt, response)

    return {
        "prompt": prompt,
        "response": response,
        "max_length": request.max_length,
        "temperature": request.temperature
    }

# --- Root endpoint para teste rápido ---
@app.get("/")
async def root():
    return {"message": "API Fei PENTA rodando com modelo fake realista!"}
