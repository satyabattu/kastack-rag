"""
app.py — Streamlit chatbot
Uses pre-built artifacts from pipeline.py
"""

import json
import os
import pickle
import re
import sys

import faiss
import numpy as np
import streamlit as st
import anthropic

# ── Setup ────────────────────────────────────────────────────────────────────

ARTIFACTS = os.environ.get("ARTIFACTS_DIR", "./artifacts")

st.set_page_config(page_title="ConvoRAG Chatbot", page_icon="🤖", layout="centered")

@st.cache_resource
def load_artifacts():
    segments = json.load(open(f"{ARTIFACTS}/topic_segments.json"))
    checkpoints = json.load(open(f"{ARTIFACTS}/checkpoints_100.json"))
    persona = json.load(open(f"{ARTIFACTS}/persona.json"))
    index = faiss.read_index(f"{ARTIFACTS}/faiss.index")
    with open(f"{ARTIFACTS}/doc_store.pkl", "rb") as f:
        docs = pickle.load(f)
    with open(f"{ARTIFACTS}/vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return segments, checkpoints, persona, index, docs, vec

def retrieve(query, index, docs, vec, top_k=5):
    q_vec = vec.transform([query]).toarray().astype("float32")
    norm = np.linalg.norm(q_vec) + 1e-10
    q_vec = q_vec / norm
    D, I = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(docs):
            results.append({**docs[idx], "score": float(score)})
    return results

def answer(query, index, docs, vec, persona, segments, checkpoints):
    client = anthropic.Anthropic()
    retrieved = retrieve(query, index, docs, vec, top_k=5)

    context_parts = []
    for r in retrieved:
        context_parts.append(
            f"[{r['type'].upper()} {r['id']} | msgs {r['start_idx']}–{r['end_idx']}]\n{r['summary']}"
        )

    context = "\n\n".join(context_parts)
    persona_str = json.dumps(persona, indent=2)

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": (
                f"You are an assistant analyzing a user's conversation history.\n\n"
                f"User Persona (extracted from conversations):\n{persona_str}\n\n"
                f"Retrieved Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer concisely based on the evidence above."
            )
        }]
    )
    return resp.content[0].text.strip(), retrieved

# ── UI ───────────────────────────────────────────────────────────────────────

st.title("🤖 ConvoRAG Chatbot")
st.caption("Ask anything about the user's conversations, habits, or personality.")

try:
    segments, checkpoints, persona, index, docs, vec = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Failed to load artifacts: {e}\nRun pipeline.py first.")
    artifacts_loaded = False

if artifacts_loaded:
    # Sidebar: persona snapshot
    with st.sidebar:
        st.header("📋 User Persona")
        if "parse_error" in persona:
            st.text(persona.get("raw", ""))
        else:
            for key, val in persona.items():
                st.subheader(key.replace("_", " ").title())
                if isinstance(val, list):
                    for item in val:
                        st.write(f"• {item}")
                elif isinstance(val, dict):
                    for k, v in val.items():
                        st.write(f"**{k}:** {v}")
                else:
                    st.write(val)

        st.divider()
        st.metric("Topic Segments", len(segments))
        st.metric("100-msg Checkpoints", len(checkpoints))
        st.metric("Index Docs", len(docs))

        st.divider()
        st.subheader("💡 Try asking:")
        suggestions = [
            "What kind of person is this user?",
            "What are their habits?",
            "How do they talk?",
            "What topics do they discuss most?",
            "Do they have any hobbies?",
        ]
        for s in suggestions:
            if st.button(s, use_container_width=True):
                st.session_state.suggested = s

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle sidebar suggestion
    if "suggested" in st.session_state:
        prompt = st.session_state.pop("suggested")
    else:
        prompt = st.chat_input("Ask about the user...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving..."):
                resp_text, retrieved = answer(prompt, index, docs, vec, persona, segments, checkpoints)
            st.markdown(resp_text)

            with st.expander("📚 Retrieved Context"):
                for r in retrieved:
                    st.markdown(f"**{r['type'].upper()} {r['id']}** (msgs {r['start_idx']}–{r['end_idx']}) — score: {r['score']:.3f}")
                    st.caption(r["summary"])

        st.session_state.messages.append({"role": "assistant", "content": resp_text})
