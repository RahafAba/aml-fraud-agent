"""Prints the known-true answers from the seeded graph, for the Cypher eval."""
from src.database_conn import DatabaseManager


def main():
    db = DatabaseManager()

    # How many customers are sanctioned or PEP? (matches the SQL answer of 27)
    r1 = db.execute_cypher(
        "MATCH (c:Customer) WHERE c.is_sanctioned = 1 OR c.is_pep = 1 "
        "RETURN count(c) AS n"
    )
    print("Sanctioned-or-PEP customers:", r1[0]["n"])

    # How many shared-device links exist?
    r2 = db.execute_cypher(
        "MATCH (:Customer)-[r:SHARES_DEVICE]->(:Customer) RETURN count(r) AS n"
    )
    print("SHARES_DEVICE relationships:", r2[0]["n"])

    # How many accounts sit in a structured-amount transfer loop (a ring)?
    r3 = db.execute_cypher(
        "MATCH path = (a:Account)-[:SENT*3..6]->(a) "
        "WHERE ALL(rel IN relationships(path) "
        "WHERE rel.amount >= 34000 AND rel.amount < 40000) "
        "RETURN count(DISTINCT a) AS n"
    )
    print("Accounts in a structured ring:", r3[0]["n"])

    db.close()


if __name__ == "__main__":
    main()