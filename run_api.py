from fastapi import FastAPI
from geb_1_3b import modeling_geblm 

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World", "Status": "API VIVO!"}

# Você pode adicionar de volta o /generate depois de escolher a API de terceiros.