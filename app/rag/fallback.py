from typing import Dict, Any, List
from ..models.schemas import QueryType, RetrievalResult
import structlog

logger = structlog.get_logger()


class FallbackStrategy:
    
    @staticmethod
    async def execute(
        query: str,
        query_type: QueryType,
        retrieval_results: List[RetrievalResult],
        confidence: float,
        sql_retriever=None
    ) -> Dict[str, Any]:
        logger.warning(
            "fallback_triggered",
            query_type=query_type.value,
            confidence=confidence,
            results_count=len(retrieval_results)
        )
        
        if confidence < 0.3 and sql_retriever:
            return await FallbackStrategy._sql_fallback(query, query_type, sql_retriever)
        elif confidence < 0.5:
            return FallbackStrategy._suggestion_fallback(query, query_type)
        else:
            return FallbackStrategy._partial_fallback(query, query_type, retrieval_results)
    
    @staticmethod
    async def _sql_fallback(query: str, query_type: QueryType, sql_retriever) -> Dict[str, Any]:
        logger.info("executing_sql_fallback", query_type=query_type.value)
        
        try:
            if query_type == QueryType.BALANCE:
                results = await sql_retriever.get_account_balances()
            elif query_type == QueryType.SPENDING:
                results = await sql_retriever.get_recent_transactions(limit=10)
            elif query_type == QueryType.SUBSCRIPTIONS:
                results = await sql_retriever.get_subscriptions()
            else:
                results = []
            
            if results:
                return {
                    "answer": f"Encontrei {len(results)} registros relacionados à sua consulta. Confira os detalhes abaixo:",
                    "retrieval_results": results,
                    "confidence": 0.4,
                    "fallback_used": "sql_direct"
                }
        except Exception as e:
            logger.error("sql_fallback_failed", error=str(e))
        
        return FallbackStrategy._generic_fallback(query)
    
    @staticmethod
    def _suggestion_fallback(query: str, query_type: QueryType) -> Dict[str, Any]:
        suggestions = {
            QueryType.SPENDING: [
                "Quanto gastei com comida este mês?",
                "Mostre meus gastos com transporte",
                "Quanto gastei na última semana?"
            ],
            QueryType.BALANCE: [
                "Qual meu saldo atual?",
                "Quanto tenho disponível?",
                "Mostre o saldo das minhas contas"
            ],
            QueryType.SUBSCRIPTIONS: [
                "Quais minhas assinaturas ativas?",
                "Quanto pago de Netflix?",
                "Liste minhas recorrências"
            ],
            QueryType.CATEGORIES: [
                "Quais categorias mais gasto?",
                "Distribua meus gastos por categoria",
                "Mostre ranking de categorias"
            ]
        }
        
        query_suggestions = suggestions.get(query_type, [
            "Quanto gastei este mês?",
            "Qual meu saldo?",
            "Quais minhas assinaturas?"
        ])
        
        return {
            "answer": (
                f"Desculpe, não consegui entender completamente sua pergunta: '{query}'. "
                f"Tente reformular de forma mais específica. Exemplos:\n\n" +
                "\n".join(f"• {s}" for s in query_suggestions)
            ),
            "retrieval_results": [],
            "confidence": 0.2,
            "fallback_used": "suggestions",
            "suggestions": query_suggestions
        }
    
    @staticmethod
    def _partial_fallback(
        query: str,
        query_type: QueryType,
        retrieval_results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        return {
            "answer": (
                f"Encontrei algumas informações relacionadas, mas não tenho total certeza "
                f"se respondem completamente sua pergunta. Aqui está o que encontrei:"
            ),
            "retrieval_results": retrieval_results,
            "confidence": 0.5,
            "fallback_used": "partial",
            "warning": "Resposta com confiança moderada. Considere reformular a pergunta."
        }
    
    @staticmethod
    def _generic_fallback(query: str) -> Dict[str, Any]:
        return {
            "answer": (
                "Desculpe, não consegui processar sua consulta no momento. "
                "Isso pode acontecer se:\n"
                "• A pergunta está muito vaga ou complexa\n"
                "• Não há dados suficientes no sistema\n"
                "• Houve um erro técnico temporário\n\n"
                "Tente reformular de forma mais simples ou específica."
            ),
            "retrieval_results": [],
            "confidence": 0.1,
            "fallback_used": "generic_error"
        }
