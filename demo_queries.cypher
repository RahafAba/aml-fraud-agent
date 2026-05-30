// demo_queries.cypher
// Ready-made AML investigation queries for Neo4j Browser (http://localhost:7474).
// Run these to explore the seeded fraud patterns and to capture screenshots.
// Tip: keep result sets small so the visualization stays readable, not a hairball.

// ---------------------------------------------------------------------------
// 1. MONEY-LAUNDERING RING
// A closed loop of transfers: money leaves an account and returns to it after
// passing through several others. 
// The amount filter targets the seeded structured (~34k-39.5k) ring transfers,
// which keeps the result a small, clean loop rather than a large incidental one.
// ---------------------------------------------------------------------------
MATCH p=(a:Account)-[r:SENT*3..6]->(a)
WHERE ALL(rel IN r WHERE rel.amount >= 34000 AND rel.amount < 40000)
RETURN p
LIMIT 1;

// ---------------------------------------------------------------------------
// 2. SYNTHETIC-IDENTITY CLUSTER
// "Different" customers quietly sharing the same device. A tight cluster here
// is a classic synthetic-identity signal.
// ---------------------------------------------------------------------------
MATCH p=(c1:Customer)-[:SHARES_DEVICE]-(c2:Customer)
RETURN p
LIMIT 25;

// ---------------------------------------------------------------------------
// 3. SANCTIONS EXPOSURE (HOP DISTANCE)
// Accounts within two hops of a sanctioned customer's account. Shows how funds
// move outward from a flagged entity.
// ---------------------------------------------------------------------------
MATCH (c:Customer {is_sanctioned: 1})-[:OWNS]->(src:Account)
MATCH p=(src)-[:SENT*1..2]->(dst:Account)
RETURN p
LIMIT 30;

// ---------------------------------------------------------------------------
// 4. HIGHEST-RISK CUSTOMERS AND THEIR ACCOUNTS
// A quick portfolio view of the riskiest customers and what they own.
// ---------------------------------------------------------------------------
MATCH (c:Customer)-[:OWNS]->(a:Account)
WHERE c.risk_score >= 80
RETURN c, a
LIMIT 40;

// ---------------------------------------------------------------------------
// 5. STRUCTURING (TEXT-METRIC CROSS-CHECK)
// Transfers sized just under the SAR 40,000 reporting threshold. Useful to confirm the
// structuring pattern exists in the graph as well as the SQL side.
// ---------------------------------------------------------------------------
MATCH (s:Account)-[t:SENT]->(d:Account)
WHERE t.amount >= 34000 AND t.amount < 40000
RETURN s, t, d
LIMIT 30;
