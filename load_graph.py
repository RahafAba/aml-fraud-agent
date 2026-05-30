"""Builds the AML knowledge graph in Neo4j directly from the SQL database.

Unlike free-text extraction, an AML graph is constructed deterministically from
structured records, which is how real systems work:

  (:Customer)-[:OWNS]->(:Account)
  (:Account)-[:SENT {amount, date}]->(:Account)
  (:Customer)-[:SHARES_DEVICE]->(:Customer)
  (:Customer)-[:SHARES_PHONE]->(:Customer)

This lets the agent answer relationship questions (rings, hops to sanctioned
entities, shared-identity clusters) that are impractical in pure SQL.
"""
import sqlite3

from src.database_conn import DatabaseManager


def load(sqlite_path: str, db: DatabaseManager):
    sql = sqlite3.connect(sqlite_path)
    sql.row_factory = sqlite3.Row
    cur = sql.cursor()

    # Clean slate plus constraints for fast MERGE.
    db.execute_cypher("MATCH (n) DETACH DELETE n")
    db.execute_cypher(
        "CREATE CONSTRAINT cust_id IF NOT EXISTS "
        "FOR (c:Customer) REQUIRE c.id IS UNIQUE"
    )
    db.execute_cypher(
        "CREATE CONSTRAINT acct_id IF NOT EXISTS "
        "FOR (a:Account) REQUIRE a.id IS UNIQUE"
    )

    # Customers
    custs = [dict(r) for r in cur.execute("SELECT * FROM customers")]
    db.execute_cypher(
        """
        UNWIND $rows AS r
        MERGE (c:Customer {id: r.customer_id})
        SET c.name = r.full_name, c.country = r.country,
            c.is_pep = r.is_pep, c.is_sanctioned = r.is_sanctioned,
            c.risk_score = r.risk_score, c.phone = r.phone,
            c.device_id = r.device_id
        """,
        {"rows": custs},
    )

    # Accounts and ownership
    accts = [dict(r) for r in cur.execute("SELECT * FROM accounts")]
    db.execute_cypher(
        """
        UNWIND $rows AS r
        MERGE (a:Account {id: r.account_id})
        SET a.type = r.account_type
        WITH a, r
        MATCH (c:Customer {id: r.customer_id})
        MERGE (c)-[:OWNS]->(a)
        """,
        {"rows": accts},
    )

    # Transactions (money flow edges)
    txns = [dict(r) for r in cur.execute("SELECT * FROM transactions")]
    db.execute_cypher(
        """
        UNWIND $rows AS r
        MATCH (s:Account {id: r.src_account})
        MATCH (d:Account {id: r.dst_account})
        MERGE (s)-[t:SENT {txn_id: r.txn_id}]->(d)
        SET t.amount = r.amount, t.date = r.txn_date, t.channel = r.channel
        """,
        {"rows": txns},
    )

    # Shared-device edges (synthetic-ID signal), derived in SQL, written as edges.
    shared_dev = [dict(r) for r in cur.execute(
        """
        SELECT a.customer_id AS c1, b.customer_id AS c2
        FROM customers a JOIN customers b
          ON a.device_id = b.device_id AND a.customer_id < b.customer_id
        """
    )]
    db.execute_cypher(
        """
        UNWIND $rows AS r
        MATCH (a:Customer {id: r.c1}), (b:Customer {id: r.c2})
        MERGE (a)-[:SHARES_DEVICE]->(b)
        """,
        {"rows": shared_dev},
    )

    shared_phone = [dict(r) for r in cur.execute(
        """
        SELECT a.customer_id AS c1, b.customer_id AS c2
        FROM customers a JOIN customers b
          ON a.phone = b.phone AND a.customer_id < b.customer_id
        """
    )]
    db.execute_cypher(
        """
        UNWIND $rows AS r
        MATCH (a:Customer {id: r.c1}), (b:Customer {id: r.c2})
        MERGE (a)-[:SHARES_PHONE]->(b)
        """,
        {"rows": shared_phone},
    )

    sql.close()
    print(f"Graph loaded: {len(custs)} customers, {len(accts)} accounts, "
          f"{len(txns)} transactions, {len(shared_dev)} shared-device links.")


if __name__ == "__main__":
    db = DatabaseManager()
    try:
        load("enterprise.db", db)
    finally:
        db.close()
    print("Done loading AML graph.")
