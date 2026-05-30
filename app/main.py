import sys
import os

import streamlit as st

# Make the project root importable so `src` resolves.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database_conn import DatabaseManager
from src.agent_logic import EnterpriseAIAgent

st.set_page_config(page_title="AML Fraud-Network Agent", layout="wide")

st.title("AML Fraud-Network Agent")
st.markdown(
    "Ask questions in plain English. The agent routes each one to SQL "
    "(metrics, risk scores), a Neo4j graph (rings, sanctions hops, shared "
    "identities), or both."
)

with st.sidebar:
    st.subheader("Try asking")
    st.markdown(
        "- How many customers are flagged as sanctioned or PEP?\n"
        "- List transactions between SAR 34,000 and SAR 40,000.\n"
        "- Find accounts that send money in a loop (laundering rings).\n"
        "- Which customers share a device or phone?\n"
        "- Which accounts are within 2 hops of a sanctioned customer?"
    )


@st.cache_resource
def get_agent():
    """Initialize backend connections once and reuse across reruns."""
    db = DatabaseManager()
    return EnterpriseAIAgent(db)


try:
    agent = get_agent()
except Exception as e:
    st.error(
        "Could not connect to the databases. Check your .env and that "
        f"Neo4j is running.\n\nDetails: {e}"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about customers, transactions, or networks..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Routing query and searching..."):
            response = agent.route_and_query(user_query)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
