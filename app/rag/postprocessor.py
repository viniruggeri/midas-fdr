from typing import List, Dict, Any
import json

from ..models.schemas import QueryType, RetrievalResult


class ResponsePostprocessor:    
    def __init__(self):
        self.response_templates = {
            QueryType.SPENDING: self._format_spending_response,
            QueryType.SUBSCRIPTIONS: self._format_subscriptions_response,
            QueryType.BALANCE: self._format_balance_response,
            QueryType.CATEGORIES: self._format_categories_response,
            QueryType.GOALS: self._format_goals_response,
            QueryType.GENERAL: self._format_general_response
        }
    
    async def format_response(
        self, 
        original_query: str,
        query_type: QueryType,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        
        if not retrieval_results:
            return "Desculpe, não encontrei informações relevantes para sua consulta."
        

        formatter = self.response_templates.get(query_type, self._format_general_response)
        return formatter(original_query, retrieval_results)
    
    def _format_spending_response(self, query: str, results: List[RetrievalResult]) -> str:
        """Formata resposta para consultas de gastos"""
        if not results:
            return "Não encontrei informações sobre seus gastos."
        

        total_amount = 0
        transactions = []
        
        for result in results:
            content = result.content
            if "R$" in content:
                try:
                    amount_str = content.split("R$")[1].split()[0].replace(",", ".")
                    amount = float(amount_str)
                    total_amount += amount
                    transactions.append(content)
                except:
                    transactions.append(content)
        
        if total_amount > 0:
            response = f"Com base nos seus dados, você gastou aproximadamente R$ {total_amount:.2f}. "
            if len(transactions) <= 3:
                response += "Principais transações: " + "; ".join(transactions[:3])
            else:
                response += f"Encontrei {len(transactions)} transações relacionadas."
        else:
            response = "Encontrei as seguintes informações sobre seus gastos: " + "; ".join(transactions[:3])
        
        return response
    
    def _format_subscriptions_response(self, query: str, results: List[RetrievalResult]) -> str:
        if not results:
            return "Não encontrei assinaturas ativas."
        
        subscriptions = []
        total_monthly = 0
        
        for result in results:
            subscriptions.append(result.content)
            if "R$" in result.content and "/mês" in result.content:
                try:
                    amount_str = result.content.split("R$")[1].split("/")[0].replace(",", ".")
                    total_monthly += float(amount_str)
                except:
                    pass
        
        response = f"Suas assinaturas ativas: {', '.join(subscriptions)}. "
        if total_monthly > 0:
            response += f"Total mensal: R$ {total_monthly:.2f}."
        
        return response
    
    def _format_balance_response(self, query: str, results: List[RetrievalResult]) -> str:
        if not results:
            return "Não foi possível acessar informações de saldo."
        
        # Return the most relevant balance info
        return results[0].content
    
    def _format_categories_response(self, query: str, results: List[RetrievalResult]) -> str:
        if not results:
            return "Não encontrei gastos nesta categoria."
        
        categories = {}
        for result in results:
            category = result.metadata.get("category", "outros")
            if category not in categories:
                categories[category] = []
            categories[category].append(result.content)
        
        response = "Gastos por categoria: "
        for category, items in categories.items():
            response += f"{category}: {len(items)} transações; "
        
        return response.rstrip("; ")
    
    def _format_goals_response(self, query: str, results: List[RetrievalResult]) -> str:
        if not results:
            return "Não encontrei informações sobre suas metas financeiras."
        
        goals_info = []
        for result in results:
            goals_info.append(result.content)
        
        return "Suas metas financeiras: " + "; ".join(goals_info)
    
    def _format_general_response(self, query: str, results: List[RetrievalResult]) -> str:
        if not results:
            return "Não encontrei informações específicas para sua consulta."
        
        # Combine most relevant results
        top_results = results[:3]
        response_parts = [result.content for result in top_results]
        
        return "Com base nos seus dados: " + "; ".join(response_parts)