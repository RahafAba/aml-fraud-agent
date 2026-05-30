"""Generates a realistic mock AML dataset in SQLite (enterprise.db).

Tables:
  customers     - account holders with KYC attributes
  accounts      - bank accounts owned by customers
  transactions  - money movement between accounts

Fraud patterns are deliberately seeded so the graph and queries have something
real to surface:
  1. Money-laundering rings: layering loops of transfers between a ring of accounts
  2. Sanctions/PEP exposure: some customers flagged; others transact within a few hops
  3. Synthetic identities: clusters of customers sharing phone/address/device
"""
import sqlite3
import random
from datetime import datetime, timedelta

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

N_CUSTOMERS = 520
N_ACCOUNTS = 640
N_TRANSACTIONS = 9000

conn = sqlite3.connect("enterprise.db")
cur = conn.cursor()

cur.executescript(
    """
    DROP TABLE IF EXISTS transactions;
    DROP TABLE IF EXISTS accounts;
    DROP TABLE IF EXISTS customers;

    CREATE TABLE customers (
        customer_id   INTEGER PRIMARY KEY,
        full_name     TEXT NOT NULL,
        country       TEXT NOT NULL,
        phone         TEXT,
        address       TEXT,
        device_id     TEXT,
        is_pep        INTEGER NOT NULL DEFAULT 0,
        is_sanctioned INTEGER NOT NULL DEFAULT 0,
        risk_score    INTEGER NOT NULL DEFAULT 0,
        onboarded_on  TEXT NOT NULL
    );

    CREATE TABLE accounts (
        account_id    INTEGER PRIMARY KEY,
        customer_id   INTEGER NOT NULL,
        account_type  TEXT NOT NULL,
        opened_on     TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );

    CREATE TABLE transactions (
        txn_id        INTEGER PRIMARY KEY,
        src_account   INTEGER NOT NULL,
        dst_account   INTEGER NOT NULL,
        amount        REAL NOT NULL,
        currency      TEXT NOT NULL DEFAULT 'SAR',
        txn_date      TEXT NOT NULL,
        channel       TEXT NOT NULL,
        FOREIGN KEY (src_account) REFERENCES accounts(account_id),
        FOREIGN KEY (dst_account) REFERENCES accounts(account_id)
    );
    """
)

countries = ["USA", "UK", "UAE", "Singapore", "Germany", "Cyprus",
             "Panama", "Nigeria", "Russia", "Switzerland", "India", "Brazil"]
high_risk = {"Cyprus", "Panama", "Russia", "Nigeria"}

start = datetime(2024, 1, 1)


def rand_date(within_days=500):
    return (start + timedelta(days=random.randint(0, within_days))).strftime("%Y-%m-%d")


# Customers
customers = []
for cid in range(1, N_CUSTOMERS + 1):
    country = random.choice(countries)
    is_pep = 1 if random.random() < 0.04 else 0
    is_sanctioned = 1 if random.random() < 0.02 else 0
    base_risk = random.randint(0, 40)
    if country in high_risk:
        base_risk += 25
    if is_pep:
        base_risk += 15
    if is_sanctioned:
        base_risk += 40
    risk = min(base_risk, 100)
    customers.append([
        cid, fake.name(), country, fake.phone_number(),
        fake.address().replace("\n", ", "), f"DEV-{fake.uuid4()[:8]}",
        is_pep, is_sanctioned, risk, rand_date(),
    ])

# Pattern 3: synthetic-identity clusters share phone/address/device.
n_clusters = 8
for _ in range(n_clusters):
    members = random.sample(range(N_CUSTOMERS), random.randint(3, 6))
    shared_phone = fake.phone_number()
    shared_addr = fake.address().replace("\n", ", ")
    shared_dev = f"DEV-{fake.uuid4()[:8]}"
    for idx in members:
        customers[idx][3] = shared_phone
        customers[idx][4] = shared_addr
        customers[idx][5] = shared_dev
        customers[idx][8] = min(customers[idx][8] + 20, 100)

cur.executemany(
    "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?)", customers
)

# Accounts and ownership
accounts = []
acct_owner = {}
for aid in range(1, N_ACCOUNTS + 1):
    owner = random.randint(1, N_CUSTOMERS)
    acct_owner[aid] = owner
    accounts.append([
        aid, owner, random.choice(["checking", "savings", "business", "crypto"]),
        rand_date(),
    ])
cur.executemany("INSERT INTO accounts VALUES (?,?,?,?)", accounts)

all_accts = list(acct_owner.keys())

# Transactions
txns = []
tid = 1

# Pattern 1: money-laundering rings, closed loops with structured amounts.
n_rings = 6
ring_accounts = set()
for _ in range(n_rings):
    ring = random.sample(all_accts, random.randint(4, 7))
    ring_accounts.update(ring)
    for _round in range(random.randint(3, 6)):
        for i in range(len(ring)):
            src = ring[i]
            dst = ring[(i + 1) % len(ring)]
            amount = round(random.uniform(34000, 39500), 2)
            txns.append([tid, src, dst, amount, "SAR", rand_date(),
                         random.choice(["wire", "ach"])])
            tid += 1

# Pattern 2: sanctioned/PEP accounts moving money outward.
sanctioned_custs = [c[0] for c in customers if c[7] == 1 or c[6] == 1]
sanctioned_accts = [a for a, o in acct_owner.items() if o in sanctioned_custs]
for sa in sanctioned_accts:
    for _ in range(random.randint(2, 5)):
        dst = random.choice(all_accts)
        txns.append([tid, sa, dst, round(random.uniform(8000, 200000), 2),
                     "SAR", rand_date(), random.choice(["wire", "swift"])])
        tid += 1

# Background noise: ordinary random transactions.
while tid <= N_TRANSACTIONS:
    src, dst = random.sample(all_accts, 2)
    txns.append([tid, src, dst, round(random.uniform(40, 28000), 2),
                 "SAR", rand_date(),
                 random.choice(["card", "ach", "wire", "transfer"])])
    tid += 1

cur.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?)", txns)

conn.commit()

print("Customers:", cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0])
print("  PEP:", cur.execute("SELECT COUNT(*) FROM customers WHERE is_pep=1").fetchone()[0])
print("  Sanctioned:", cur.execute("SELECT COUNT(*) FROM customers WHERE is_sanctioned=1").fetchone()[0])
print("Accounts:", cur.execute("SELECT COUNT(*) FROM accounts").fetchone()[0])
print("Transactions:", cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])
print("Ring accounts seeded:", len(ring_accounts))
conn.close()
print("SQL database 'enterprise.db' initialized with seeded AML patterns.")
