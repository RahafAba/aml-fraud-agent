# AML Fraud Network Agent: Graph-RAG and Text-to-SQL

A natural language agent for **anti-money-laundering (AML) investigation**. Ask a question in English and it routes to the right backend: a SQL database for metrics and risk scores, a **Neo4j graph** for network relationships (laundering rings, sanctions exposure, synthetic-identity clusters), or both. Runs fully locally with no API keys via Ollama, using a **fine-tuned 3B model** for Cypher generation.

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
| LLM (local) | Ollama running a fine-tuned `llama3.2:3b` |
| Orchestration | LangChain |
| Structured data | SQLite + SQLAlchemy (Text-to-SQL) |
| Graph data | Neo4j (Cypher) |
| Fine-tuning | Unsloth + TRL (QLoRA), GGUF export |
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

## Evaluation

The agent has two failure surfaces: routing the question to the right backend, and generating a valid query. There is a labelled eval for each, runnable locally with no GPU.

### Routing (`run_routing_eval.py`)

20 labelled questions across SQL / GRAPH / HYBRID.

| Route | Result |
|-------|--------|
| SQL | 8/8 correct |
| GRAPH | 8/8 correct |
| HYBRID | 0/4 — every hybrid question routed to GRAPH |

Overall: **16/20 (80%)**. The model handles pure-SQL and pure-graph questions perfectly but never recognises a question that needs both a network traversal and a metric. This is the single systematic routing weakness.

### Cypher generation (`run_cypher_eval.py`)

6 questions run against the seeded graph, scored three ways: did it run, did it return rows, and (for questions with a known answer) did it return the right count. Ground-truth answers come from `eval_ground_truth.py`.

## Fine-tuning: Text-to-Cypher

The base 3B model produces invalid Cypher on complex multi-hop queries — hallucinated `exists(path)`, `distinct(x, y)`, SQL `SELECT ... FROM` inside Cypher, and stacked `RETURN` clauses instead of `UNION`. Prompt rules reduced but did not eliminate these.

I fine-tuned `Llama-3.2-3B-Instruct` with QLoRA (Unsloth, free Colab T4 GPU) on ~530 synthetic schema-specific Text-to-Cypher pairs (`finetune/`), then quantised to GGUF and loaded it back into Ollama. After a first round, I did error analysis on the remaining failures, added targeted examples for the failing patterns (UNION, aggregation, ring + amount filter), and retrained.

### Results (same 6-question Cypher eval)

| Metric | Base 3B | Fine-tune v1 | Fine-tune v2 |
|--------|---------|--------------|--------------|
| Valid (ran) | 3/6 | 3/6 | **5/6** |
| Non-empty (rows) | 2/6 | 3/6 | **5/6** |
| Exact match | 0/1 | 1/1 | **2/2** |

The multi-hop ring query the base model failed every time now generates valid Cypher. The one remaining failure is an over-generalisation (fusing a column projection with an aggregate), a known limit of small-model fine-tuning rather than a missing pattern.

### Reproduce the fine-tune

1. `python finetune/generate_dataset.py` — build the training pairs
2. Run `finetune/train_cypher_lora.py` on a Colab T4 GPU (upload the data first)
3. Download the resulting GGUF, register it with Ollama using the `Modelfile`
4. Set `LOCAL_MODEL = "cypher-tuned"` in `src/agent_logic.py`, then run the evals

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A running [Neo4j](https://neo4j.com/download/) instance

### Setup
```bash
# 1. Local model (no API key, no cost)
ollama pull llama3.2:3b

# 2. Install
git clone https://github.com/RahafAba/aml-fraud-agent.git
cd aml-fraud-agent
pip install -r requirements.txt

# 3. Configure
cp .env.example .env # edit with your Neo4j credentials

# 4. Build the data (SQL first, then graph reads from it)
python init_sql_db.py # generates enterprise.db with seeded patterns
python load_graph.py # builds the Neo4j graph from that data

# 5. (Optional) load the fine-tuned Cypher model into Ollama
ollama create cypher-tuned -f Modelfile

# 6. Run
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
├── finetune/
│   ├── generate_dataset.py  # builds synthetic Text-to-Cypher training pairs
│   ├── train_cypher_lora.py # QLoRA fine-tune (Colab T4), exports GGUF
│   ├── cypher_train.jsonl   # training data
│   └── cypher_train_val.jsonl
├── init_sql_db.py           # generates the mock AML SQLite database
├── load_graph.py            # builds the Neo4j graph from the SQL data
├── eval_questions.py        # labelled routing questions
├── run_routing_eval.py      # routing accuracy + confusion matrix
├── eval_ground_truth.py     # known-true answers from the seeded graph
├── run_cypher_eval.py       # Cypher validity / correctness eval
├── Modelfile                # loads the fine-tuned GGUF into Ollama
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
- **Measured** Routing and Cypher generation are both evaluated against labelled sets, and the fine-tune is justified by before/after numbers rather than a claim.

## Limitations

- Mock data; this is an architecture and skills demonstration, not a production AML system.
- The router does not yet detect HYBRID questions (combined network + metric) reliably, a known weakness surfaced by the routing eval.
- The fine-tuned Cypher model still fails on some compositional queries (fusing a projection with an aggregate); this is a documented limit. 
