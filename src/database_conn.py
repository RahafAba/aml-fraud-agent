import os
import re
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Only a single leading SELECT or WITH...SELECT statement is permitted.
_SELECT_ONLY = re.compile(r"^\s*(with\b.*?\bselect\b|select\b)", re.IGNORECASE | re.DOTALL)
# Reject anything that smuggles in a second statement or a write keyword.
_FORBIDDEN = re.compile(
    r";\s*\S|\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|merge|call|attach|pragma|copy)\b",
    re.IGNORECASE,
)


class DatabaseManager:
    def __init__(self):
        # SQL connection. Prefer a URL whose DB user has read-only grants.
        self.sql_uri = os.getenv("SQL_DATABASE_URL", "sqlite:///enterprise.db")
        self.sql_engine = create_engine(self.sql_uri, pool_pre_ping=True)

        # Neo4j connection.
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.graph_driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
            max_connection_pool_size=20,
        )
        # Fail fast if the graph is unreachable.
        self.graph_driver.verify_connectivity()

    def _assert_read_only(self, query: str) -> None:
        """Block non-SELECT SQL even if the DB user is mis-provisioned."""
        if not _SELECT_ONLY.match(query):
            raise ValueError("Only SELECT/WITH queries are allowed.")
        if _FORBIDDEN.search(query):
            raise ValueError("Query contains a forbidden keyword or multiple statements.")

    def execute_sql(self, query: str):
        """Executes a read-only SQL query and returns a list of dict rows."""
        self._assert_read_only(query)
        try:
            with self.sql_engine.connect() as connection:
                result = connection.execute(text(query))
                return [dict(row._mapping) for row in result]
        except SQLAlchemyError as e:
            raise RuntimeError(f"SQL execution failed: {e}") from e

    def execute_cypher(self, query: str, parameters=None):
        """Executes a Cypher query and returns a list of dict records.
        The list is materialized inside the session so it stays valid after close."""
        with self.graph_driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def close(self):
        self.sql_engine.dispose()
        self.graph_driver.close()
