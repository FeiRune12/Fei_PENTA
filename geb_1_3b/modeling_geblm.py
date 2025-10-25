# geb_1_3b/modeling_geblm.py

class GEBLMForCausalLM:
    """
    Mock/fake da classe GEBLMForCausalLM para testes da API
    sem precisar do modelo real.
    """

    @classmethod
    def from_pretrained(cls, model_name, trust_remote_code=False):
        print(f"Carregando modelo fake {model_name} (trust_remote_code={trust_remote_code})")
        return cls()

    def to(self, device):
        print(f"Movendo modelo fake para {device}")
        return self

    def generate(self, prompt, max_length=100, temperature=1.0, **kwargs):
        # Respostas pré-definidas para testes
        prompt_lower = prompt.lower()
        if "olá" in prompt_lower:
            return "Olá! Como posso ajudar você hoje?"
        elif "ajuda" in prompt_lower:
            return "Claro! Estou aqui para te auxiliar."
        elif "teste" in prompt_lower:
            return "Este é um teste do modelo fake. Tudo funcionando!"
        else:
            # Resposta genérica simulando parâmetros do modelo real
            return f"[RESPOSTA SIMULADA] Prompt: {prompt}, max_length: {max_length}, temperature: {temperature}"

    def __call__(self, prompt, **kwargs):
        # Permite chamar o objeto como função
        return self.generate(prompt, **kwargs)
