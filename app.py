import os
from typing import List, Tuple

import streamlit as st
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


DB_DIR = "./db"
EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "llama3"
TOP_K = 8
MAX_CONTEXT_DOCS = 3
MAX_DISTANCE = 1.1


def load_vectorstore() -> Chroma:
    if not os.path.exists(DB_DIR) or not any(os.scandir(DB_DIR)):
        raise FileNotFoundError(
            f"Vector DB not found at '{DB_DIR}'. Pehle index build karo."
        )

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


def keyword_overlap_score(question: str, text: str) -> float:
    q_words = set(question.lower().split())
    t_words = set(text.lower().split())

    if not q_words or not t_words:
        return 0.0

    overlap = q_words.intersection(t_words)
    return len(overlap) / max(len(q_words), 1)


def retrieve_docs(question: str, vectorstore: Chroma) -> List[Tuple[Document, float]]:
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)

    ranked: List[Tuple[Document, float]] = []

    for doc, distance in results:
        content = doc.page_content or ""
        overlap_bonus = keyword_overlap_score(question, content)

        # lower distance is better
        adjusted_score = distance - (overlap_bonus * 0.15)
        ranked.append((doc, adjusted_score))

    ranked.sort(key=lambda x: x[1])

    filtered = [(doc, score) for doc, score in ranked if score <= MAX_DISTANCE]

    if not filtered:
        return []

    return filtered[:MAX_CONTEXT_DOCS]


def format_context(docs_with_scores: List[Tuple[Document, float]]) -> str:
    if not docs_with_scores:
        return "No context found."

    chunks = []

    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        title = doc.metadata.get("title", "unknown")
        page = doc.metadata.get("page", "unknown")
        section = doc.metadata.get("section", "unknown")
        heading = doc.metadata.get("heading", "unknown")
        category = doc.metadata.get("category", "general")
        source = doc.metadata.get("source", "unknown")

        chunk_text = (
            f"[Source {i}]\n"
            f"Title: {title}\n"
            f"Page: {page}\n"
            f"Section: {section}\n"
            f"Heading: {heading}\n"
            f"Category: {category}\n"
            f"URL: {source}\n"
            f"Content: {doc.page_content}"
        )
        chunks.append(chunk_text)

    return "\n\n".join(chunks)



def build_prompt(question: str, context: str) -> str:
    return f"""
You are an Exactink website assistant.

Use only the context below.
Do not add outside knowledge.
If the answer is not clearly supported by the context, reply exactly:
I could not find this information in the provided data

Rules:
- Keep answer short
- Use simple Hinglish
- Prefer direct bullets only if needed
- Mention only the most relevant source URLs at the end
- Do not mention any information not present in context


Context:
{context}

Question:
{question}

Answer:
""".strip()


def direct_business_faq(question: str):
    q = question.lower()

    if "address" in q or "office address" in q:
        return "Exactink address is 3824 Cedar Springs Rd #656, Dallas, TX 75219."

    if "india office" in q or "indore" in q:
        return "Exactink also lists Indore, Madhya Pradesh, India as a location."

    if "email" in q or "contact email" in q:
        return "Exactink contact email is inquiry@exactink.com."

    return None

def ask_bot(question: str, vectorstore: Chroma, llm: ChatOllama):
    docs_with_scores = retrieve_docs(question, vectorstore)

    if not docs_with_scores:
        return "I could not find this information in the provided data", []

    context = format_context(docs_with_scores)
    prompt = build_prompt(question, context)

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return answer, docs_with_scores


@st.cache_resource
def init_system():
    vectorstore = load_vectorstore()
    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0,
    )
    return vectorstore, llm


st.set_page_config(page_title="Exactink RAG Chatbot", page_icon="🤖", layout="wide")

st.title(" Exactink Chatbot")

with st.sidebar:
    st.header("Settings")
    st.write(f"**Embedding Model:** {EMBED_MODEL}")
    st.write(f"**Chat Model:** {CHAT_MODEL}")
    st.write(f"**DB Path:** {DB_DIR}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    debug_mode = st.checkbox("Show Debug Info", value=False)

try:
    vectorstore, llm = init_system()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            st.markdown("**Sources:**")
            for src in message["sources"]:
                st.markdown(f"- {src}")

        if debug_mode and message.get("debug_chunks"):
            with st.expander("Debug Chunks"):
                for idx, chunk in enumerate(message["debug_chunks"], start=1):
                    st.markdown(f"**Chunk {idx}**")
                    st.write(f"Score: {chunk['score']}")
                    st.write(f"Heading: {chunk['heading']}")
                    st.write(f"Source: {chunk['source']}")
                    st.write(chunk["preview"])

question = st.chat_input("Ask a question ...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking...."):
            try:
                answer, docs_with_scores = ask_bot(question, vectorstore, llm)

                sources = []
                debug_chunks = []
                seen = set()

                for doc, score in docs_with_scores:
                    src = doc.metadata.get("source", "unknown")
                    heading = doc.metadata.get("heading", "unknown")
                    preview = doc.page_content[:400]

                    if src not in seen:
                        seen.add(src)
                        sources.append(src)

                    debug_chunks.append(
                        {
                            "score": round(score, 4),
                            "heading": heading,
                            "source": src,
                            "preview": preview,
                        }
                    )

                st.markdown(answer)

                if sources:
                    st.markdown("**Sources:**")
                    for src in sources:
                        st.markdown(f"- {src}")

                if debug_mode and debug_chunks:
                    with st.expander("Debug Chunks"):
                        for idx, chunk in enumerate(debug_chunks, start=1):
                            st.markdown(f"**Chunk {idx}**")
                            st.write(f"Score: {chunk['score']}")
                            st.write(f"Heading: {chunk['heading']}")
                            st.write(f"Source: {chunk['source']}")
                            st.write(chunk["preview"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "debug_chunks": debug_chunks,
                    }
                )

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "debug_chunks": [],
                    }
                )