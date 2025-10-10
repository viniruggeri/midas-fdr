from typing import Dict
from ..models.schemas import QueryType


class HybridRetrievalConfig:
    
    WEIGHTS = {
        QueryType.SPENDING: {
            "faiss": 0.4,
            "tfidf": 0.3,
            "sql": 0.3
        },
        QueryType.BALANCE: {
            "faiss": 0.2,
            "tfidf": 0.1,
            "sql": 0.7
        },
        QueryType.SUBSCRIPTIONS: {
            "faiss": 0.5,
            "tfidf": 0.4,
            "sql": 0.1
        },
        QueryType.CATEGORIES: {
            "faiss": 0.45,
            "tfidf": 0.45,
            "sql": 0.1
        },
        QueryType.GOALS: {
            "faiss": 0.4,
            "tfidf": 0.3,
            "sql": 0.3
        },
        QueryType.GENERAL: {
            "faiss": 0.4,
            "tfidf": 0.3,
            "sql": 0.3
        }
    }
    
    CONFIDENCE_THRESHOLD = {
        QueryType.SPENDING: 0.6,
        QueryType.BALANCE: 0.7,
        QueryType.SUBSCRIPTIONS: 0.65,
        QueryType.CATEGORIES: 0.55,
        QueryType.GOALS: 0.6,
        QueryType.GENERAL: 0.5
    }
    
    TOP_K = {
        "faiss": 5,
        "tfidf": 5,
        "sql": 3
    }
    
    @classmethod
    def get_weights(cls, query_type: QueryType) -> Dict[str, float]:
        return cls.WEIGHTS.get(query_type, cls.WEIGHTS[QueryType.GENERAL])
    
    @classmethod
    def get_threshold(cls, query_type: QueryType) -> float:
        return cls.CONFIDENCE_THRESHOLD.get(query_type, 0.5)


QUERY_SYNONYMS = {
    "gastos": ["despesas", "débitos", "saídas", "pagamentos"],
    "ganhos": ["receitas", "entradas", "créditos", "recebimentos"],
    "transferências": ["movimentações", "envios", "remessas"],
    "conta": ["banco", "instituição financeira", "cartão"],
    "cartão": ["cartão de crédito", "cartão de débito", "card"],
    "comida": ["alimentação", "refeições", "delivery", "restaurante"],
    "transporte": ["uber", "99", "combustível", "estacionamento"],
    "lazer": ["entretenimento", "diversão", "streaming"],
    "assinaturas": ["recorrentes", "mensalidades", "subscriptions"],
    "mês": ["mensal", "mes", "mês passado", "este mês"],
    "ano": ["anual", "ano passado", "este ano"],
    "semana": ["semanal", "esta semana", "semana passada"],
}


def expand_query_with_synonyms(query: str, max_expansions: int = 3) -> str:
    words = query.lower().split()
    expanded = []
    
    for word in words:
        expanded.append(word)
        if word in QUERY_SYNONYMS:
            synonyms = QUERY_SYNONYMS[word][:max_expansions]
            expanded.extend(synonyms)
    
    return " ".join(expanded)
