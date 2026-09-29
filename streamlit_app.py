import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/chat"


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("🤖 Agentic AI RAG Assistant")

st.markdown(
    """
    Ask questions about the **Agentic AI eBook**.

    This assistant retrieves relevant content from the eBook
    using **Pinecone** and generates answers using **LangGraph + Gemini**.
    """
)

st.divider()


# ------------------------------------------------------------
# QUESTION INPUT
# ------------------------------------------------------------

query = st.text_area(
    "Ask a question",
    placeholder="Example: What is Agentic AI according to the eBook?",
    height=100,
)


ask_button = st.button(
    "🔍 Ask Question",
    type="primary",
)


# ------------------------------------------------------------
# API CALL
# ------------------------------------------------------------

if ask_button:

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching the eBook and generating an answer..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure it is running with:\n\n"
                    "`uvicorn app:app --reload`"
                )

            except requests.exceptions.RequestException as e:

                st.error(f"API request failed: {e}")

            else:

                # ------------------------------------------------
                # ANSWER
                # ------------------------------------------------

                st.subheader("Answer")

                st.write(data["final_answer"])


                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                confidence = data["confidence_score"]

                st.subheader("Confidence")

                st.progress(confidence)

                st.write(f"**{confidence:.2f} ({confidence * 100:.0f}%)**")


                # ------------------------------------------------
                # RETRIEVED CONTEXT
                # ------------------------------------------------

                st.subheader("Retrieved Context")

                chunks = data["retrieved_context_chunks"]

                for i, chunk in enumerate(chunks, start=1):

                    with st.expander(f"Context Chunk {i}"):

                        st.text(chunk)