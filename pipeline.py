"""
pipeline.py
-----------
Processes conversations.csv and builds:
  - Topic checkpoints (segment summaries)
  - 100-message checkpoints
  - User persona (JSON)
  - FAISS index for retrieval
"""

import csv
import json
import os
import pickle
import re
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

# ─── 1. Load & flatten messages ─────────────────────────────────────────────

def load_messages(csv_path: str) -> list[dict]:
    """
    Each CSV row = one conversation (one 'day').
    We flatten into a global chronological message list.
    Returns list of {global_idx, conv_idx, msg_idx, speaker, text}
    """
    messages = []
    global_idx = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for conv_idx, row in enumerate(reader):
            if not row:
                continue
            conv_text = row[0]
            lines = [l.strip() for l in conv_text.split("\n") if l.strip()]
            for msg_idx, line in enumerate(lines):
                m = re.match(r"^(User \d+):\s*(.+)$", line)
                if m:
                    speaker = m.group(1)
                    text = m.group(2)
                else:
                    speaker = "Unknown"
                    text = line
                messages.append({
                    "global_idx": global_idx,
                    "conv_idx": conv_idx,
                    "msg_idx": msg_idx,
                    "speaker": speaker,
                    "text": text,
                })
                global_idx += 1
    print(f"Loaded {len(messages)} messages from {conv_idx+1} conversations.")
    return messages


# ─── 2. Topic change detection ───────────────────────────────────────────────

WINDOW = 5          # messages to represent "current" context
STRIDE = 5          # check every N messages
TOPIC_THRESHOLD = 0.25  # cosine similarity drop = topic change


def detect_topic_segments(messages: list[dict]) -> list[dict]:
    """
    Sliding window cosine similarity over TF-IDF vectors.
    When similarity drops below threshold → new topic segment.
    Returns list of {segment_id, start_idx, end_idx, messages}
    """
    texts = [m["text"] for m in messages]

    # Build TF-IDF on all texts
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf = vec.fit_transform(texts)

    segments = []
    seg_start = 0
    seg_id = 0
    prev_vec = None

    for i in range(0, len(messages), STRIDE):
        window_end = min(i + WINDOW, len(messages))
        chunk_vecs = tfidf[i:window_end]
        # mean vector for this window
        curr_vec = np.asarray(chunk_vecs.mean(axis=0))

        if prev_vec is not None:
            sim = cosine_similarity(prev_vec, curr_vec)[0][0]
            if sim < TOPIC_THRESHOLD and i - seg_start >= 10:
                # topic change detected
                segments.append({
                    "segment_id": seg_id,
                    "start_idx": seg_start,
                    "end_idx": i - 1,
                    "messages": messages[seg_start:i],
                })
                seg_id += 1
                seg_start = i

        prev_vec = curr_vec

    # last segment
    segments.append({
        "segment_id": seg_id,
        "start_idx": seg_start,
        "end_idx": len(messages) - 1,
        "messages": messages[seg_start:],
    })

    print(f"Detected {len(segments)} topic segments.")
    return segments


# ─── 3. 100-message checkpoints ──────────────────────────────────────────────

def build_100_checkpoints(messages: list[dict]) -> list[dict]:
    checkpoints = []
    for i in range(0, len(messages), 100):
        chunk = messages[i:i+100]
        checkpoints.append({
            "checkpoint_id": i // 100,
            "start_idx": i,
            "end_idx": min(i + 99, len(messages) - 1),
            "messages": chunk,
        })
    print(f"Built {len(checkpoints)} 100-message checkpoints.")
    return checkpoints


# ─── 4. Summarize via Claude ──────────────────────────────────────────────────

def summarize_chunk(msgs: list[dict], label: str) -> str:
    """Call Claude to summarize a list of messages."""
    convo_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in msgs[:80])  # cap at 80 msgs
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"Summarize the following conversation segment concisely (2-4 sentences). "
                f"Label: {label}\n\n{convo_text}"
            )
        }]
    )
    return resp.content[0].text.strip()


def summarize_all(segments: list[dict], checkpoints: list[dict],
                  max_segs=None, max_cps=None):
    """Add summaries to segments and checkpoints in-place."""
    segs_to_do = segments[:max_segs] if max_segs else segments
    cps_to_do = checkpoints[:max_cps] if max_cps else checkpoints

    print(f"Summarizing {len(segs_to_do)} topic segments...")
    for s in segs_to_do:
        label = f"Topic {s['segment_id']+1} (msgs {s['start_idx']}–{s['end_idx']})"
        s["summary"] = summarize_chunk(s["messages"], label)
        if s["segment_id"] % 10 == 0:
            print(f"  segment {s['segment_id']} done")

    print(f"Summarizing {len(cps_to_do)} 100-msg checkpoints...")
    for cp in cps_to_do:
        label = f"100-msg checkpoint {cp['checkpoint_id']} (msgs {cp['start_idx']}–{cp['end_idx']})"
        cp["summary"] = summarize_chunk(cp["messages"], label)
        if cp["checkpoint_id"] % 10 == 0:
            print(f"  checkpoint {cp['checkpoint_id']} done")


# ─── 5. Persona extraction ───────────────────────────────────────────────────

def extract_persona(messages: list[dict]) -> dict:
    """
    Sample messages evenly across the full conversation history
    and ask Claude to extract a structured persona.
    """
    # Sample ~200 messages spread across all conversations
    step = max(1, len(messages) // 200)
    sample = messages[::step][:200]
    convo_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in sample)

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": (
                "You are analyzing conversation data to build a user persona.\n"
                "From the messages below, extract a persona for 'User 1' ONLY.\n"
                "Return ONLY valid JSON with these keys:\n"
                "  habits: list of strings\n"
                "  personal_facts: list of strings\n"
                "  personality_traits: list of strings\n"
                "  communication_style: object with keys: "
                "    avg_message_length (short/medium/long), tone, emoji_usage, formality\n"
                "Base everything on actual signals in the text.\n\n"
                f"{convo_text}"
            )
        }]
    )
    raw = resp.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        persona = json.loads(raw)
    except json.JSONDecodeError:
        persona = {"raw": raw, "parse_error": True}
    return persona


# ─── 6. FAISS index ──────────────────────────────────────────────────────────

def build_faiss_index(segments: list[dict], checkpoints: list[dict]) -> tuple:
    """
    Build a FAISS index over all summaries.
    Returns (index, doc_store, vectorizer)
    doc_store: list of {type, id, summary, start_idx, end_idx}
    """
    docs = []
    for s in segments:
        if "summary" in s:
            docs.append({
                "type": "topic",
                "id": s["segment_id"],
                "summary": s["summary"],
                "start_idx": s["start_idx"],
                "end_idx": s["end_idx"],
            })
    for cp in checkpoints:
        if "summary" in cp:
            docs.append({
                "type": "checkpoint",
                "id": cp["checkpoint_id"],
                "summary": cp["summary"],
                "start_idx": cp["start_idx"],
                "end_idx": cp["end_idx"],
            })

    summaries = [d["summary"] for d in docs]
    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    mat = vec.fit_transform(summaries).toarray().astype("float32")

    # L2-normalize for cosine sim via inner product
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    mat = mat / norms

    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    print(f"FAISS index built: {index.ntotal} docs, dim={mat.shape[1]}")
    return index, docs, vec


# ─── 7. Save / Load artifacts ────────────────────────────────────────────────

def save_artifacts(base_dir: str, segments, checkpoints, persona, index, docs, vec):
    os.makedirs(base_dir, exist_ok=True)
    # Save summaries only (not full message lists — too large)
    seg_summaries = [{k: v for k, v in s.items() if k != "messages"} for s in segments]
    cp_summaries = [{k: v for k, v in c.items() if k != "messages"} for c in checkpoints]

    with open(f"{base_dir}/topic_segments.json", "w") as f:
        json.dump(seg_summaries, f, indent=2)
    with open(f"{base_dir}/checkpoints_100.json", "w") as f:
        json.dump(cp_summaries, f, indent=2)
    with open(f"{base_dir}/persona.json", "w") as f:
        json.dump(persona, f, indent=2)

    faiss.write_index(index, f"{base_dir}/faiss.index")
    with open(f"{base_dir}/doc_store.pkl", "wb") as f:
        pickle.dump(docs, f)
    with open(f"{base_dir}/vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    print(f"Artifacts saved to {base_dir}/")


def load_artifacts(base_dir: str):
    with open(f"{base_dir}/topic_segments.json") as f:
        segments = json.load(f)
    with open(f"{base_dir}/checkpoints_100.json") as f:
        checkpoints = json.load(f)
    with open(f"{base_dir}/persona.json") as f:
        persona = json.load(f)
    index = faiss.read_index(f"{base_dir}/faiss.index")
    with open(f"{base_dir}/doc_store.pkl", "rb") as f:
        docs = pickle.load(f)
    with open(f"{base_dir}/vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return segments, checkpoints, persona, index, docs, vec


# ─── 8. Retrieval + Answer generation ────────────────────────────────────────

def retrieve(query: str, index, docs, vec, top_k=5) -> list[dict]:
    q_vec = vec.transform([query]).toarray().astype("float32")
    norm = np.linalg.norm(q_vec) + 1e-10
    q_vec = q_vec / norm
    D, I = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(docs):
            results.append({**docs[idx], "score": float(score)})
    return results


def answer_query(query: str, index, docs, vec, persona: dict, messages: list[dict]) -> str:
    retrieved = retrieve(query, index, docs, vec, top_k=5)

    context_parts = []
    for r in retrieved:
        context_parts.append(
            f"[{r['type'].upper()} {r['id']} | msgs {r['start_idx']}-{r['end_idx']}]\n{r['summary']}"
        )
        # Grab a few raw messages from this segment for extra grounding
        chunk = messages[r['start_idx']:min(r['start_idx']+5, r['end_idx']+1)]
        for m in chunk:
            context_parts.append(f"  {m['speaker']}: {m['text']}")

    context = "\n\n".join(context_parts)
    persona_str = json.dumps(persona, indent=2)

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"You are an assistant with access to a user's conversation history.\n\n"
                f"User Persona:\n{persona_str}\n\n"
                f"Retrieved Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer based on the context and persona. Be direct and specific."
            )
        }]
    )
    return resp.content[0].text.strip()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    BASE = "."
    ARTIFACTS = f"{BASE}/artifacts"
    CSV = f"{BASE}/conversations.csv"

    print("=== Loading messages ===")
    messages = load_messages(CSV)

    print("\n=== Topic segmentation ===")
    segments = detect_topic_segments(messages)

    print("\n=== 100-message checkpoints ===")
    checkpoints = build_100_checkpoints(messages)

    # For demo: summarize first 50 segments + first 20 checkpoints
    # Full run would do all — adjust as needed
    MAX_SEGS = int(os.environ.get("MAX_SEGS", 50))
    MAX_CPS = int(os.environ.get("MAX_CPS", 20))

    print(f"\n=== Summarizing (MAX_SEGS={MAX_SEGS}, MAX_CPS={MAX_CPS}) ===")
    summarize_all(segments, checkpoints, max_segs=MAX_SEGS, max_cps=MAX_CPS)

    print("\n=== Persona extraction ===")
    persona = extract_persona(messages)
    print(json.dumps(persona, indent=2))

    print("\n=== Building FAISS index ===")
    index, docs, vec = build_faiss_index(segments, checkpoints)

    print("\n=== Saving artifacts ===")
    save_artifacts(ARTIFACTS, segments, checkpoints, persona, index, docs, vec)

    print("\nDone. Run app.py to start the chatbot.")
