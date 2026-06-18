"""Cypher correctness eval: valid? non-empty? exact-match on known answers?"""
from src.database_conn import DatabaseManager
from src.agent_logic import EnterpriseAIAgent

# question -> known-true answer (from eval_ground_truth.py). None = no exact check.
CASES = [
    ("How many customers are sanctioned or PEP?", 27),
    ("How many pairs of customers share a device?", 49),
    ("How many accounts are in a money-laundering ring "
     "(a loop of structured transfers between 34000 and 40000)?", 29),
    ("Which customers share a device or phone?", None),
    ("Show accounts within 2 hops of a sanctioned customer.", None),
    ("Find money-laundering rings where accounts send money in a loop.", None),
]


def main():
    agent = EnterpriseAIAgent(DatabaseManager())
    valid = nonempty = exact = exact_checked = 0

    for question, truth in CASES:
        cypher = agent.generate_cypher(question)
        print(f"\nQ: {question}")
        print(f"   Cypher: {cypher}")
        try:
            rows = agent.db.execute_cypher(cypher)
            valid += 1
            if rows:
                nonempty += 1
            print(f"   -> ran OK, {len(rows)} row(s)")
            if truth is not None:
                exact_checked += 1
                # grab the single number the query returned, if any
                got = None
                if rows and len(rows[0]) == 1:
                    got = list(rows[0].values())[0]
                if got == truth:
                    exact += 1
                    print(f"   -> EXACT match: {got} == {truth}")
                else:
                    print(f"   -> got {got}, expected {truth}")
        except Exception as e:
            print(f"   -> INVALID: {e}")

    n = len(CASES)
    print("\n" + "=" * 40)
    print(f"Valid (ran without error): {valid}/{n}")
    print(f"Non-empty (returned rows): {nonempty}/{n}")
    print(f"Exact match (known answers): {exact}/{exact_checked}")


if __name__ == "__main__":
    main()