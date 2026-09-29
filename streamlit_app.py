import os
import streamlit as st

# Load Streamlit Cloud secrets into environment variables.
# Locally, the existing .env file will continue to work.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

    if "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

    if "PINECONE_INDEX_NAME" in st.secrets:
        os.environ["PINECONE_INDEX_NAME"] = st.secrets["PINECONE_INDEX_NAME"]

except Exception:
    # When running locally, use the .env file instead.
    pass


from src.graph import graph


st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI RAG Assistant")

st.markdown(
    """
Ask questions about the **Agentic AI eBook**.

This assistant retrieves relevant content from the eBook
using **Pinecone** and generates answers using
**LangGraph + Gemini**.
"""
)

st.divider()


query = st.text_area(
    "Ask a question",
    placeholder="Example: What is Agentic AI according to the eBook?",
    height=100,
)

ask_button = st.button("🔍 Ask Question", type="primary")


if ask_button:

    if not query.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching the eBook and generating an answer..."):

            try:
                result = graph.invoke(
                    {
                        "question": query,
                        "context": [],
                        "answer": "",
                        "score": 0.0,
                    }
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

            else:

                st.subheader("Answer")

                st.write(result["answer"])

                st.subheader("Confidence")

                confidence = result["score"]

                st.progress(confidence)

                st.write(
                    f"**{confidence:.2f} ({confidence * 100:.0f}%)**"
                )

                st.subheader("Retrieved Context")

                chunks = result["context"]

                if chunks:

                    for i, chunk in enumerate(chunks, start=1):

                        with st.expander(
                            f"Context Chunk {i}"
                        ):
                            st.text(chunk)

                else:

                    st.info(
                        "No relevant context was retrieved from the eBook."
                    )