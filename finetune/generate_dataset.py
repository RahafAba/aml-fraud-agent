"""
generate_dataset.py
-------------------
Generate synthetic (natural-language question -> Neo4j Cypher) pairs for the
AML Fraud-Network Agent, matched to the project's ACTUAL schema:

  (:Customer {id, name, country, is_pep, is_sanctioned, risk_score})
  (:Account {id, type})
  (:Customer)-[:OWNS]->(:Account)
  (:Account)-[:SENT {amount, date, channel}]->(:Account)
  (:Customer)-[:SHARES_DEVICE]->(:Customer)
  (:Customer)-[:SHARES_PHONE]->(:Customer)

The target Cypher for every example is hand-verified valid syntax, so the model
learns correct patterns -- especially the multi-hop ring queries the base model
keeps getting wrong (exists(path), distinct(x,y), WHERE id IN (path), etc.).

Output: JSONL with {"instruction", "input", "output"}.
"""

import json
import random
import argparse
from pathlib import Path

random.seed(42)

SCHEMA = """
(:Customer {id, name, country, is_pep, is_sanctioned, risk_score})
(:Account {id, type})
(:Customer)-[:OWNS]->(:Account)
(:Account)-[:SENT {amount, date, channel}]->(:Account)
(:Customer)-[:SHARES_DEVICE]->(:Customer)
(:Customer)-[:SHARES_PHONE]->(:Customer)
""".strip()

INSTRUCTION = (
    "Convert the question into a single valid read-only Neo4j Cypher query "
    "(MATCH/RETURN only). Use only the labels, relationship types and properties "
    "in this schema. Return only the Cypher.\n\nSchema:\n" + SCHEMA
)

COUNTRIES = ["USA", "UK", "UAE", "Singapore", "Germany", "Cyprus",
             "Panama", "Nigeria", "Russia", "Switzerland", "India", "Brazil"]
TYPES = ["checking", "savings", "business", "crypto"]
CHANNELS = ["wire", "ach", "card", "transfer", "swift"]
SCORES = [40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
HOPS = [(2, 4), (2, 5), (2, 6), (3, 5), (3, 6), (4, 6), (3, 7)]
LIMITS = [5, 10, 20, 25, 30, 50, 100]
AMOUNTS = [25000, 50000, 75000, 100000, 150000, 200000, 250000, 500000]


def s():
    lo, hi = random.choice(HOPS)
    return {
        "country": random.choice(COUNTRIES),
        "type": random.choice(TYPES),
        "channel": random.choice(CHANNELS),
        "score": random.choice(SCORES),
        "lo": lo, "hi": hi,
        "limit": random.choice(LIMITS),
        "amount": random.choice(AMOUNTS),
        "h": random.choice([1, 2, 3]),
    }


# Each template returns (list of paraphrases, cypher). All cypher is valid syntax.

def t_pep_or_sanctioned(_):
    q = ["How many customers are flagged as sanctioned or PEP?",
         "Count the customers who are PEP or sanctioned.",
         "How many customers are either sanctioned or politically exposed?"]
    c = ("MATCH (c:Customer) WHERE c.is_sanctioned = 1 OR c.is_pep = 1 "
         "RETURN count(c) AS n")
    return q, c


def t_sanctioned_list(_):
    q = ["List all sanctioned customers.",
         "Show me every customer who is sanctioned.",
         "Which customers are flagged as sanctioned?"]
    c = "MATCH (c:Customer {is_sanctioned: 1}) RETURN c.id, c.name, c.country"
    return q, c


def t_high_risk(v):
    q = [f"Which customers have a risk score above {v['score']}?",
         f"List customers with risk score over {v['score']}.",
         f"Show high-risk customers scoring more than {v['score']}."]
    c = (f"MATCH (c:Customer) WHERE c.risk_score > {v['score']} "
         f"RETURN c.id, c.name, c.risk_score ORDER BY c.risk_score DESC")
    return q, c


def t_by_country(v):
    q = [f"List customers from {v['country']}.",
         f"Show every customer in {v['country']}.",
         f"Which customers are based in {v['country']}?"]
    c = (f"MATCH (c:Customer {{country: '{v['country']}'}}) "
         f"RETURN c.id, c.name, c.risk_score")
    return q, c


def t_accounts_per_type(v):
    q = [f"List all {v['type']} accounts.",
         f"Show accounts of type {v['type']}.",
         f"Which accounts are {v['type']} accounts?"]
    c = f"MATCH (a:Account {{type: '{v['type']}'}}) RETURN a.id"
    return q, c


def t_owns(_):
    q = ["Which accounts does each customer own?",
         "Show customers and the accounts they own.",
         "List ownership links between customers and accounts."]
    c = ("MATCH (c:Customer)-[:OWNS]->(a:Account) "
         "RETURN c.id, c.name, a.id, a.type")
    return q, c


def t_shared_device(_):
    q = ["Which customers share a device?",
         "Find pairs of customers using the same device.",
         "Show shared-device links between customers."]
    c = ("MATCH (c1:Customer)-[:SHARES_DEVICE]->(c2:Customer) "
         "RETURN c1.id, c1.name, c2.id, c2.name")
    return q, c


def t_shared_phone(_):
    q = ["Which customers share a phone number?",
         "Find customers using the same phone.",
         "Show shared-phone links between customers."]
    c = ("MATCH (c1:Customer)-[:SHARES_PHONE]->(c2:Customer) "
         "RETURN c1.id, c1.name, c2.id, c2.name")
    return q, c


def t_count_shared_device(_):
    q = ["How many pairs of customers share a device?",
         "Count the shared-device relationships.",
         "How many SHARES_DEVICE links are there?"]
    c = ("MATCH (:Customer)-[r:SHARES_DEVICE]->(:Customer) "
         "RETURN count(r) AS n")
    return q, c


def t_hops_from_sanctioned(v):
    h = v['h']
    q = [f"Show accounts within {h} hops of a sanctioned customer.",
         f"Which accounts are reachable from a sanctioned customer in {h} hops?",
         f"Find accounts up to {h} hops from a sanctioned entity."]
    c = (f"MATCH (c:Customer {{is_sanctioned: 1}})-[:OWNS]->(src:Account) "
         f"MATCH path = (src)-[:SENT*1..{h}]->(dst:Account) "
         f"RETURN DISTINCT dst.id LIMIT {v['limit']}")
    return q, c


def t_ring(v):
    lo, hi = v['lo'], v['hi']
    q = ["Find money-laundering rings where accounts send money in a loop.",
         "Detect closed transfer loops between accounts.",
         f"Find accounts that send money in a cycle of {lo} to {hi} hops."]
    c = (f"MATCH path = (a:Account)-[:SENT*{lo}..{hi}]->(a) "
         f"RETURN DISTINCT a.id LIMIT {v['limit']}")
    return q, c


def t_ring_structured(v):
    q = ["Find laundering rings using structured amounts just under 40000.",
         "Detect transfer loops where every hop is between 34000 and 40000.",
         "Show rings of structured transfers (34000 to 40000)."]
    c = ("MATCH path = (a:Account)-[:SENT*3..6]->(a) "
         "WHERE ALL(rel IN relationships(path) "
         "WHERE rel.amount >= 34000 AND rel.amount < 40000) "
         "RETURN DISTINCT a.id LIMIT 25")
    return q, c


def t_count_ring(_):
    q = ["How many accounts are in a money-laundering ring?",
         "Count the accounts that sit in a structured transfer loop.",
         "How many accounts belong to a structured ring (34000 to 40000)?"]
    c = ("MATCH path = (a:Account)-[:SENT*3..6]->(a) "
         "WHERE ALL(rel IN relationships(path) "
         "WHERE rel.amount >= 34000 AND rel.amount < 40000) "
         "RETURN count(DISTINCT a) AS n")
    return q, c


def t_high_value_sent(v):
    amt = v['amount']
    q = [f"Show transfers above {amt}.",
         f"Find SENT transactions over {amt}.",
         f"Which transfers exceed {amt}?"]
    c = (f"MATCH (a:Account)-[t:SENT]->(b:Account) WHERE t.amount > {amt} "
         f"RETURN a.id, b.id, t.amount ORDER BY t.amount DESC LIMIT {v['limit']}")
    return q, c


def t_structuring_range(_):
    q = ["List transfers between 34000 and 40000.",
         "Show SENT transactions in the structuring range 34000 to 40000.",
         "Which transfers are sized just under the 40000 threshold?"]
    c = ("MATCH (a:Account)-[t:SENT]->(b:Account) "
         "WHERE t.amount >= 34000 AND t.amount < 40000 "
         "RETURN a.id, b.id, t.amount")
    return q, c


def t_channel(v):
    q = [f"Show {v['channel']} transfers.",
         f"List transactions sent by {v['channel']}.",
         f"Which transfers used the {v['channel']} channel?"]
    c = (f"MATCH (a:Account)-[t:SENT {{channel: '{v['channel']}'}}]->(b:Account) "
         f"RETURN a.id, b.id, t.amount")
    return q, c


def t_sanctioned_outflow(_):
    q = ["Show sanctioned customers and the total they transferred out.",
         "For each sanctioned customer, sum the money sent from their accounts.",
         "Total outgoing transfers per sanctioned customer."]
    c = ("MATCH (c:Customer {is_sanctioned: 1})-[:OWNS]->(a:Account)-[t:SENT]->() "
         "RETURN c.id, c.name, sum(t.amount) AS total ORDER BY total DESC")
    return q, c


def t_riskiest(v):
    q = [f"Who are the top {v['limit']} riskiest customers?",
         f"List the {v['limit']} customers with the highest risk score.",
         f"Show the {v['limit']} highest-risk customers and their accounts."]
    c = (f"MATCH (c:Customer)-[:OWNS]->(a:Account) "
         f"RETURN c.id, c.name, c.risk_score, a.id "
         f"ORDER BY c.risk_score DESC LIMIT {v['limit']}")
    return q, c


def t_pep_count_country(_):
    q = ["How many PEPs are there per country?",
         "Count politically exposed customers by country.",
         "Break down PEP customers by country."]
    c = ("MATCH (c:Customer {is_pep: 1}) "
         "RETURN c.country, count(c) AS n ORDER BY n DESC")
    return q, c


# ---------------------------------------------------------------------------
# COMPOSITIONAL templates -- target the three failure modes found in eval:
#   (1) count + grouping   (2) ring + amount filter   (3) device OR phone (UNION)
# These were underrepresented in v1, so the model composed them wrong.
# ---------------------------------------------------------------------------

def t_device_or_phone_union(v):
    # The model wrote two stacked RETURNs; correct Cypher uses UNION.
    # Vary the returned columns so this yields multiple distinct examples.
    cols = random.choice([
        ("c1.id, c2.id", "c1.id, c2.id"),
        ("c1.id, c1.name, c2.id, c2.name", "c1.id, c1.name, c2.id, c2.name"),
        ("c1.name, c2.name", "c1.name, c2.name"),
        ("DISTINCT c1.id, c2.id", "DISTINCT c1.id, c2.id"),
    ])
    q = ["Which customers share a device or phone?",
         "Find customers who share either a device or a phone number.",
         "Show customers linked by a shared device or shared phone.",
         "List customers connected by a shared device or phone."]
    c = (f"MATCH (c1:Customer)-[:SHARES_DEVICE]->(c2:Customer) "
         f"RETURN {cols[0]} "
         f"UNION "
         f"MATCH (c1:Customer)-[:SHARES_PHONE]->(c2:Customer) "
         f"RETURN {cols[1]}")
    return q, c


def t_count_device_pairs(_):
    # The model garbled the RETURN ... COUNT. Correct grouping/count form.
    q = ["How many pairs of customers share a device?",
         "Count how many customer pairs share a device.",
         "What is the number of shared-device pairs?"]
    c = ("MATCH (c1:Customer)-[:SHARES_DEVICE]->(c2:Customer) "
         "RETURN count(*) AS n")
    return q, c


def t_count_phone_pairs(_):
    q = ["How many pairs of customers share a phone?",
         "Count the customer pairs sharing a phone number.",
         "What is the number of shared-phone pairs?"]
    c = ("MATCH (c1:Customer)-[:SHARES_PHONE]->(c2:Customer) "
         "RETURN count(*) AS n")
    return q, c


def t_count_ring_filtered(_):
    # The model hallucinated SQL (SELECT...FROM) inside Cypher.
    # Correct: filter the path relationships with ALL(... IN relationships(path)).
    q = ["How many accounts are in a money-laundering ring of structured transfers?",
         "Count accounts in a loop where every transfer is between 34000 and 40000.",
         "How many accounts sit in a structured ring (34000 to 40000)?"]
    c = ("MATCH path = (a:Account)-[:SENT*2..6]->(a) "
         "WHERE ALL(rel IN relationships(path) "
         "WHERE rel.amount >= 34000 AND rel.amount < 40000) "
         "RETURN count(DISTINCT a) AS n")
    return q, c


def t_ring_filtered_list(v):
    q = ["List accounts in a structured laundering ring.",
         "Show accounts in a loop of transfers between 34000 and 40000.",
         "Find ring accounts where every hop is a structured amount."]
    c = (f"MATCH path = (a:Account)-[:SENT*2..6]->(a) "
         f"WHERE ALL(rel IN relationships(path) "
         f"WHERE rel.amount >= 34000 AND rel.amount < 40000) "
         f"RETURN DISTINCT a.id LIMIT {v['limit']}")
    return q, c


def t_count_accounts_by_type(_):
    q = ["How many accounts are there of each type?",
         "Count accounts grouped by type.",
         "Break down the number of accounts by account type."]
    c = ("MATCH (a:Account) "
         "RETURN a.type, count(a) AS n ORDER BY n DESC")
    return q, c


def t_count_customers_by_country(_):
    q = ["How many customers are there per country?",
         "Count customers grouped by country.",
         "Break down customers by country."]
    c = ("MATCH (c:Customer) "
         "RETURN c.country, count(c) AS n ORDER BY n DESC")
    return q, c


def t_avg_risk_by_country(_):
    q = ["What is the average risk score per country?",
         "Show the mean risk score grouped by country.",
         "Average customer risk score by country."]
    c = ("MATCH (c:Customer) "
         "RETURN c.country, avg(c.risk_score) AS avg_risk ORDER BY avg_risk DESC")
    return q, c


def t_count_transfers_by_channel(_):
    q = ["How many transfers are there per channel?",
         "Count SENT transactions grouped by channel.",
         "Break down transfers by channel."]
    c = ("MATCH ()-[t:SENT]->() "
         "RETURN t.channel, count(t) AS n ORDER BY n DESC")
    return q, c


def t_sanctioned_or_pep_union(_):
    q = ["List customers who are sanctioned or PEP.",
         "Show every customer that is either sanctioned or a PEP.",
         "Which customers are sanctioned or politically exposed?"]
    c = ("MATCH (c:Customer) WHERE c.is_sanctioned = 1 OR c.is_pep = 1 "
         "RETURN c.id, c.name, c.is_sanctioned, c.is_pep")
    return q, c


def t_shared_any_union(v):
    # Another UNION example: count of all shared-identity links of either kind.
    q = ["How many shared-identity links are there in total (device or phone)?",
         "Count all shared-device and shared-phone links combined.",
         "Total number of shared device or phone connections."]
    c = ("MATCH (:Customer)-[r:SHARES_DEVICE]->(:Customer) "
         "RETURN count(r) AS n "
         "UNION ALL "
         "MATCH (:Customer)-[r:SHARES_PHONE]->(:Customer) "
         "RETURN count(r) AS n")
    return q, c


COMPOSITIONAL = [
    t_device_or_phone_union, t_count_device_pairs, t_count_phone_pairs,
    t_count_ring_filtered, t_ring_filtered_list, t_count_accounts_by_type,
    t_count_customers_by_country, t_avg_risk_by_country,
    t_count_transfers_by_channel, t_sanctioned_or_pep_union,
    t_shared_any_union,
]


TEMPLATES = [
    t_pep_or_sanctioned, t_sanctioned_list, t_high_risk, t_by_country,
    t_accounts_per_type, t_owns, t_shared_device, t_shared_phone,
    t_count_shared_device, t_hops_from_sanctioned, t_ring, t_ring_structured,
    t_count_ring, t_high_value_sent, t_structuring_range, t_channel,
    t_sanctioned_outflow, t_riskiest, t_pep_count_country,
]


def generate(n):
    rows, seen = [], set()
    tries = 0
    # Oversample COMPOSITIONAL: these are the patterns v1 got wrong (UNION,
    # count+group, ring+amount-filter). Drawing them ~40% of the time ensures
    # the model sees them often enough to learn the correct composition.
    while len(rows) < n and tries < n * 60:
        tries += 1
        if random.random() < 0.40:
            fn = random.choice(COMPOSITIONAL)
        else:
            fn = random.choice(TEMPLATES)
        qs, c = fn(s())
        q = random.choice(qs)
        key = (q, c)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"instruction": INSTRUCTION, "input": q, "output": c})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default="cypher_train.jsonl")
    ap.add_argument("--val-split", type=float, default=0.1)
    args = ap.parse_args()

    rows = generate(args.n)
    random.shuffle(rows)
    n_val = int(len(rows) * args.val_split)
    val, train = rows[:n_val], rows[n_val:]

    Path(args.out).write_text("\n".join(json.dumps(r) for r in train))
    Path(args.out.replace(".jsonl", "_val.jsonl")).write_text(
        "\n".join(json.dumps(r) for r in val))
    print(f"{len(rows)} unique pairs -> {len(train)} train / {len(val)} val")


if __name__ == "__main__":
    main()
