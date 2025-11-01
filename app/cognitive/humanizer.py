"""
MIDAS FDR v2 - Humanizer LLM
Tradução de ICE para linguagem natural (LLM "algemado" aos dados da ICE)
"""

import os
from openai import AsyncOpenAI
from .schemas import CognitiveOutput
import json


class HumanizerLLM:
    """
    Humanizer: traduz ICE para linguagem natural
    - LLM CONSTRANGIDO: apenas traduz dados existentes, sem inventar fatos
    - Modelo leve: gpt-4o-mini
    - Temperatura baixa: 0.3 (fidelidade aos dados)
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
    
    async def humanize(self, ice_output: CognitiveOutput) -> str:
        """
        Traduz ICE para texto humanizado
        REGRA CRÍTICA: LLM apenas reformula dados existentes na ICE, sem adicionar fatos
        """
        # Serializar ICE para contexto
        ice_json = ice_output.to_json()
        
        system_prompt = """Você é o módulo de humanização do MIDAS FDR v2.

Sua ÚNICA função é traduzir a Interface Cognitiva Estruturada (ICE) para linguagem natural fluente e compreensível.

REGRAS ABSOLUTAS:
1. Você NUNCA inventa fatos - apenas reformula os dados presentes na ICE
2. Toda afirmação DEVE ter origem explícita nos campos da ICE
3. Se a ICE não contém informação, você diz "não foi possível determinar"
4. Você menciona a cadeia de raciocínio de forma clara
5. Sempre cite a confiança final e a profundidade do raciocínio

FORMATO DE RESPOSTA:
- Comece com uma resposta direta à pergunta do usuário
- Explique o raciocínio passo a passo (inference_chain)
- Conclua com métricas de confiança e coerência do grafo

PROIBIDO:
- Adicionar informações não presentes na ICE
- Fazer suposições ou inferências próprias
- Ignorar os passos de raciocínio documentados
"""
        
        user_prompt = f"""Traduza a seguinte ICE para linguagem natural:

```json
{ice_json}
```

Pergunta original do usuário: "{ice_output.query}"

Responda de forma clara e humanizada, seguindo RIGOROSAMENTE as regras."""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        humanized_text = response.choices[0].message.content
        return humanized_text
