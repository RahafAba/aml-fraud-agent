# AML Fraud Network Agent: Graph-RAG and Text-to-SQL

A natural language agent for **anti-money-laundering (AML) investigation**. Ask a question in English and it routes to the right backend: a SQL database for metrics and risk scores, a **Neo4j graph** for network relationships (laundering rings, sanctions exposure, synthetic-identity clusters), or both. Runs fully locally with no API keys via Ollama.

---

## The Problem

Financial crime investigators face two kinds of question that no single database answers well:

- **Metric questions** such as "Which customers moved more than SAR 39,000, just under the SAR 40,000 reporting threshold?" map naturally to SQL.
- **Network questions** such as "Show me the laundering ring this account sits in" or "Which accounts are within three hops of a sanctioned entity?" map naturally to a graph. They are impractical in SQL because they require traversing many relationships.

Real AML platforms (NICE Actimize, Quantexa, Featurespace) are built around exactly this split. This project demonstrates the core pattern: one natural-language interface that decides where each question belongs and can combine both sides.

## Why a graph is actually needed here

With 500+ accounts and 9,000 transactions, the relationships are not something a human can eyeball from a table. A money-laundering ring is a closed loop `A→B→C→D→A` hidden among thousands of legitimate transfers; a synthetic-identity cluster is a set of "different" customers quietly sharing a device or phone. Finding these means traversing the network, which is what graphs are for.

## Seeded Fraud Patterns

The mock data is generated with three real AML typologies deliberately planted in the noise:

| Pattern | How it's seeded | How it's found |
|---------|-----------------|----------------|
| **Money-laundering rings** | Closed transfer loops with structured (~SAR 34k-39.5k) amounts | Graph cycle detection |
| **Sanctions / PEP exposure** | Flagged customers move funds outward | Graph hop-distance from flagged nodes |
| **Synthetic identities** | Clusters of customers share device/phone/address | `SHARES_DEVICE` / `SHARES_PHONE` edges |

## How It Works

```
            User question (plain English)
                        |
                        v
                  Routing Agent
              (classifies the query)
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
      SQL             GRAPH           HYBRID
   metrics,        rings, hops,      both,
   risk scores,    shared-identity   merged
   thresholds      clusters
        |               |               |
        v               v               |
   SQLite /          Neo4j              |
   SQLAlchemy        (Cypher)  <--------+
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python |
| LLM (local) | Ollama running `llama3.1` |
| Orchestration | LangChain |
| Structured data | SQLite + SQLAlchemy (Text-to-SQL) |
| Graph data | Neo4j (Cypher) |
| Mock data | Faker |
| UI | Streamlit |

## Example Questions

| Question | Routed to |
|----------|-----------|
| "How many customers are flagged as sanctioned or PEP?" | SQL |
| "List transactions between SAR 34,000 and SAR 40,000 (structuring)." | SQL |
| "Find money-laundering rings (accounts that send money in a loop)." | GRAPH |
| "Which customers share a device or phone with another customer?" | GRAPH |
| "Which accounts are within 2 hops of a sanctioned customer?" | GRAPH |
| "For accounts inside a laundering ring, what's their total transferred amount?" | HYBRID |

## Graph Schema

```
(:Customer {id, name, country, is_pep, is_sanctioned, risk_score})
(:Account {id, type})

(:Customer)-[:OWNS]->(:Account)
(:Account)-[:SENT {amount, date, channel}]->(:Account)
(:Customer)-[:SHARES_DEVICE]->(:Customer)
(:Customer)-[:SHARES_PHONE]->(:Customer)
```

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A running [Neo4j](https://neo4j.com/download/) instance

### Setup
```bash
# 1. Local model (no API key, no cost)
ollama pull llama3.1

# 2. Install
git clone https://github.com/RahafAba/aml-fraud-agent.git
cd aml-fraud-agent
pip install -r requirements.txt

# 3. Configure
cp .env.example .env # edit with your Neo4j credentials

# 4. Build the data (SQL first, then graph reads from it)
python init_sql_db.py # generates enterprise.db with seeded patterns
python load_graph.py # builds the Neo4j graph from that data

# 5. Run
streamlit run app/main.py
```

## Project Structure

```
.
├── src/
│   ├── database_conn.py     # SQL + Neo4j connections, read-only SQL guard
│   └── agent_logic.py       # routing agent (SQL / GRAPH / HYBRID)
├── app/
│   └── main.py              # Streamlit chat UI
├── init_sql_db.py           # generates the mock AML SQLite database
├── load_graph.py            # builds the Neo4j graph from the SQL data
├── demo_queries.cypher      # ready-made investigation queries
├── requirements.txt
├── .env.example
└── README.md
```

## Design Notes

- **Runs locally with no API cost.** The LLM is served by Ollama, and the model name is a single constant (`LOCAL_MODEL`), so swapping it is a one-line change.
- **Graph built deterministically from records.** The graph is constructed from structured data rather than free-text extraction. This is how real systems work and is far more reliable than LLM entity extraction.
- **Realistic threshold.** Structuring transactions are sized just under **SAR 40,000**, the level above which transfers are automatically reported to SAMA (Saudi Central Bank), so the laundering pattern mirrors a real regulatory trigger.
- **Read-only SQL guard.** Generated SQL is rejected unless it is a single `SELECT` or `WITH` statement, as defense-in-depth over read-only database permissions.
- **Cypher write-clause guard.** Generated Cypher is refused if it contains any write operation.
- **Live schema introspection.** The Text-to-SQL chain reads the real schema rather than a hardcoded description.

## Limitations

- Mock data; this is an architecture and skills demonstration, not a production AML system.
- Local model Cypher generation can occasionally need prompt tuning for complex multi-hop questions; `qwen2.5:14b` handles these better than smaller models.

