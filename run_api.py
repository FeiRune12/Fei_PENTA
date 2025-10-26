# Importa a classe FastAPI
from fastapi import FastAPI
# Importa seu módulo local
from geb_1_3b import modeling_geblm 

# -----------------------------------------------------------
# PASSO 1: CRIE A INSTÂNCIA DA APLICAÇÃO (O Uvicorn PROCURA POR ESTA LINHA!)
# -----------------------------------------------------------
app = FastAPI()

# -----------------------------------------------------------
# PASSO 2: CRIE SEUS ENDPOINTS (Rotas)
# -----------------------------------------------------------

@app.get("/")
def read_root():
    return {"Hello": "World", "Status": "API VIVO!"}

# Adicione suas outras rotas aqui, usando modeling_geblm se necessário.
# Exemplo:
# @app.post("/processar")
# def processar_dados(data: dict):
#     resultado = modeling_geblm.sua_funcao(data)
#     return {"resultado": resultado}