"""Labelled routing eval set. Each item: (question, correct_route)."""

QUESTIONS = [
    # --- SQL (metrics, counts, filters) ---
    ("How many customers are flagged as sanctioned or PEP?", "SQL"),
    ("What is the average risk score across all customers?", "SQL"),
    ("List transactions between SAR 34,000 and SAR 40,000.", "SQL"),
    ("Show the 10 highest-value transactions.", "SQL"),
    ("How many accounts are there of each account type?", "SQL"),
    ("Which customers have a risk score above 80?", "SQL"),
    ("Count the transactions per channel.", "SQL"),
    ("How many customers are from each country?", "SQL"),

    # --- GRAPH (relationships, rings, hops, shared identity) ---
    ("Find money-laundering rings where accounts send money in a loop.", "GRAPH"),
    ("Which customers share a device or phone?", "GRAPH"),
    ("Show accounts within 2 hops of a sanctioned customer.", "GRAPH"),
    ("Detect synthetic-identity clusters sharing the same device.", "GRAPH"),
    ("Find closed transfer loops between accounts.", "GRAPH"),
    ("Which accounts are reachable from a sanctioned entity within three hops?", "GRAPH"),
    ("Show customers connected through a shared phone number.", "GRAPH"),
    ("Find accounts that form a cycle of transfers.", "GRAPH"),

    # --- HYBRID (relationship + metric together) ---
    ("For accounts inside a laundering ring, what is their total transferred amount?", "HYBRID"),
    ("Which shared-device clusters have a combined risk score above 200?", "HYBRID"),
    ("Show sanctioned customers and the total they transferred out.", "HYBRID"),
    ("For customers within 2 hops of a sanctioned entity, what is their average risk score?", "HYBRID"),
]