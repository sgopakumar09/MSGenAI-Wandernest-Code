"""Streamlit interface for the WanderNest travel assistant."""
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search_engine import TravelSearchEngine

st.set_page_config(page_title="WanderNest Travels - AI Assistant", layout="wide")
st.title("WanderNest Travels AI Assistant")
st.markdown("Get grounded answers about travel policies and requirements.")


@st.cache_resource
def get_engine():
    return TravelSearchEngine()


def display_results(results, generated_response):
    st.subheader("Answer")
    st.markdown(generated_response)
    if results:
        with st.expander(f"Sources ({len(results)})"):
            for index, document in enumerate(results, start=1):
                source = document.metadata.get("source", "Unknown")
                category = document.metadata.get("category", "N/A")
                st.markdown(f"**{index}. {source}** — {category}")
                st.write(document.page_content[:1000])
                st.divider()
    else:
        st.info("No matching source documents were retrieved.")


with st.sidebar:
    st.header("Travel topics")
    st.markdown(
        "- Flight and baggage policies\n"
        "- Refunds and cancellations\n"
        "- Air India booking policies\n"
        "- U.S. Department of Transportation rules"
    )
    st.divider()
    if st.button("Clear cached search engine"):
        get_engine.clear()
        st.rerun()

st.markdown("### Ask a travel question")
examples = [
    "What is the refund deadline for a ticket purchased with a credit card?",
    "What is Air India's policy for changing a name on a PNR?",
    "Which countries require aircraft disinsection?",
]
columns = st.columns(3)
for column, example in zip(columns, examples):
    if column.button(example, use_container_width=True):
        st.session_state["travel_query"] = example

query = st.text_input(
    "Travel question",
    key="travel_query",
    placeholder="Ask about a travel policy or requirement...",
)

if st.button("Search", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Enter a travel question to search.")
    else:
        try:
            engine = get_engine()
            with st.spinner("Searching the travel knowledge base..."):
                results, processed_query = engine.search_by_text(query, k=5)
                if processed_query == "Query blocked by security checks.":
                    st.warning(processed_query)
                else:
                    answer = engine.synthesize_response(results, query)
                    display_results(results, answer)
        except Exception as exc:
            st.error(f"Search is unavailable: {exc}")
