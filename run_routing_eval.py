"""Run the routing eval against the agent. No GPU needed."""
from collections import Counter

from src.database_conn import DatabaseManager
from src.agent_logic import EnterpriseAIAgent
from eval_questions import QUESTIONS

ROUTES = ["SQL", "GRAPH", "HYBRID"]


def main():
    agent = EnterpriseAIAgent(DatabaseManager())

    correct = 0
    confusion = {g: Counter() for g in ROUTES}
    misroutes = []

    for question, gold in QUESTIONS:
        pred = agent.classify_route(question)
        confusion[gold][pred] += 1
        if pred == gold:
            correct += 1
        else:
            misroutes.append((question, gold, pred))

    n = len(QUESTIONS)
    print(f"\nRouting accuracy: {correct}/{n} = {correct/n:.2f}\n")

    print("Confusion matrix (rows = correct, cols = predicted):")
    print("gold\\pred".ljust(12) + "".join(r.rjust(8) for r in ROUTES))
    for g in ROUTES:
        print(g.ljust(12) + "".join(str(confusion[g][p]).rjust(8) for p in ROUTES))

    if misroutes:
        print(f"\nMisrouted ({len(misroutes)}):")
        for q, gold, pred in misroutes:
            print(f"  {gold} -> {pred}: {q}")


if __name__ == "__main__":
    main()