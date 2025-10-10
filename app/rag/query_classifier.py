from typing import Dict
from enum import Enum
import re

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

from ..models.schemas import QueryType


class QueryClassifier:
    def __init__(self):
        self.patterns = {
            QueryType.SPENDING: [
                r"gast(ei|ou|amos)",
                r"quanto.*gastar",
                r"despesa",
                r"saída",
                r"débito",
                r"compra",
                r"pagamento"
            ],
            QueryType.SUBSCRIPTIONS: [
                r"assinatura",
                r"subscription",
                r"mensalidade",
                r"recorrente",
                r"netflix|spotify|amazon|uber",
                r"mensais?"
            ],
            QueryType.BALANCE: [
                r"saldo",
                r"balance",
                r"tenho.*conta",
                r"sobrou",
                r"restou",
                r"total.*conta"
            ],
            QueryType.CATEGORIES: [
                r"categoria",
                r"tipo.*gasto",
                r"classificação",
                r"delivery|comida|transporte|lazer"
            ],
            QueryType.GOALS: [
                r"meta",
                r"objetivo",
                r"cofrinho",
                r"economia",
                r"poupar",
                r"guardar"
            ]
        }
    
    async def classify(self, query: str) -> QueryType:
        query_lower = query.lower()


        scores = {}
        for query_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[query_type] = score
        

        if max(scores.values()) > 0:
            return max(scores.keys(), key=lambda k: scores[k])
        
        return QueryType.GENERAL