import re

from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from src.database_conn import DatabaseManager

# Cypher labels and relationship types cannot be parameterized, so they must be
# validated against a strict identifier pattern before interpolation.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Single place to set the local model used across the project.
LOCAL_MODEL = "llama3.1"


def safe_ident(value: str) -> str:
    """Allow only plain identifiers for Cypher labels and relationship types."""
    if not _IDENT.match(value or ""):
        raise ValueError(f"Unsafe Cypher identifier: {value!r}")
    return value


class EnterpriseAIAgent:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        # Runs locally via Ollama. No API key, no per-call cost.
        self.llm = ChatOllama(model=LOCAL_MODEL, temperature=0)
        # Wrap the same engine so the SQL chain can introspect the real schema
        # instead of relying on a hardcoded table description.
        self.sql_db = SQLDatabase(self.db.sql_engine)
        self.sql_chain = create_sql_query_chain(self.llm, self.sql_db)

    def route_and_query(self, user_question: str) -> str:
        """Routes the question to SQL, GRAPH, or both (HYBRID)."""
        routing_prompt = (
            "You route AML/fraud-investigation questions to a backend.\n"
            "- 'SQL': aggregates and metrics over customers/accounts/transactions "
            "(counts, totals, risk scores, transactions over a threshold, PEP/sanctioned lists).\n"
            "- 'GRAPH': network relationships, money-laundering rings/loops, hops between "
            "an account and a sanctioned entity, shared-identity clusters (same device/phone).\n"
            "- 'HYBRID': needs a network relationship AND a numeric metric together.\n\n"
            f"Question: {user_question}\n"
            "Respond with exactly one word: SQL, GRAPH, or HYBRID."
        )
        route = self.llm.invoke(routing_prompt).content.strip().upper()

        if route == "SQL":
            return self._handle_text_to_sql(user_question)
        if route == "HYBRID":
            sql_part = self._handle_text_to_sql(user_question)
            graph_part = self._handle_graph_rag(user_question)
            return f"{sql_part}\n\n---\n\n{graph_part}"
        return self._handle_graph_rag(user_question)

    def _handle_text_to_sql(self, question: str) -> str:
        try:
            generated_sql = self.sql_chain.invoke({"question": question}).strip()
            data_results = self.db.execute_sql(generated_sql)
            return (
                f"**SQL Result**\nGenerated Query: `{generated_sql}`\n\n"
                f"Data: {data_results}"
            )
        except Exception as e:
            return f"**SQL Error:** {e}"

    def _handle_graph_rag(self, question: str) -> str:
        prompt = (
            "Convert this AML question into a single read-only Neo4j Cypher query "
            "(MATCH/RETURN only, no writes). Return ONLY the raw Cypher, no markdown.\n"
            "Graph schema:\n"
            "  (:Customer {id, name, country, is_pep, is_sanctioned, risk_score})\n"
            "  (:Account {id, type})\n"
            "  (:Customer)-[:OWNS]->(:Account)\n"
            "  (:Account)-[:SENT {amount, date, channel}]->(:Account)\n"
            "  (:Customer)-[:SHARES_DEVICE]->(:Customer)\n"
            "  (:Customer)-[:SHARES_PHONE]->(:Customer)\n"
            f"Question: {question}"
        )
        generated_cypher = self.llm.invoke(prompt).content.strip()
        if re.search(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP)\b", generated_cypher, re.IGNORECASE):
            return "**Graph Error:** generated query contained a write clause; refused."
        try:
            graph_results = self.db.execute_cypher(generated_cypher)
            return (
                f"**Graph Result**\nGenerated Cypher: `{generated_cypher}`\n\n"
                f"Data: {graph_results}"
            )
        except Exception as e:
            return f"**Graph Error:** {e}"
