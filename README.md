---
title: kastack-rag
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.37.0"
python_version: "3.10"
app_file: app.py
pinned: false
---
# ConvoRAG — KaStack AI/ML Intern Task


End-to-end RAG system over conversation data with topic segmentation, persona extraction, and a Streamlit chatbot.

## Architecture

```
conversations.csv
      │
      ▼
pipeline.py
  ├── load_messages()         → 191,853 messages (flattened from 11,001 conversations)
  ├── detect_topic_segments() → ~18,000 topic segments
  ├── build_100_checkpoints() → 1,919 checkpoints
  ├── summarize_all()         → Claude API summaries per segment + checkpoint
  ├── extract_persona()       → structured JSON persona
  └── build_faiss_index()     → TF-IDF + FAISS for retrieval
      │
      ▼
artifacts/
  ├── topic_segments.json
  ├── checkpoints_100.json
  ├── persona.json
  ├── faiss.index
  ├── doc_store.pkl
  └── vectorizer.pkl
      │
      ▼
app.py (Streamlit chatbot)
```

## How Topic Change Detection Works

1. A TF-IDF matrix is built over all messages (5,000 features, English stop words removed).
2. A sliding window of 5 messages moves in steps of 5 through the chronological stream.
3. The mean TF-IDF vector of each window is computed.
4. Cosine similarity between the current window and the previous window is calculated.
5. If similarity drops below **0.25** AND the current segment has ≥10 messages → **new topic segment** is created.
6. This produces segments like:
   - Topic 1 → msgs 0–47 → summary
   - Topic 2 → msgs 48–112 → summary
   - ...

Key parameters (tunable in `pipeline.py`):
- `WINDOW = 5` — context window size
- `STRIDE = 5` — check frequency
- `TOPIC_THRESHOLD = 0.25` — similarity cutoff

## How Retrieval Works

1. All topic summaries and 100-message checkpoint summaries are indexed.
2. Each summary is TF-IDF vectorized and L2-normalized.
3. Stored in a **FAISS IndexFlatIP** (inner product = cosine similarity on normalized vectors).
4. At query time: query → TF-IDF → normalize → FAISS search → top-5 docs returned.
5. Retrieved summaries + raw message snippets are injected as context into Claude.

## How Persona Is Built

1. ~200 messages are sampled evenly across all conversations.
2. Sent to Claude with a structured prompt requesting JSON output.
3. Extracted fields:
   - `habits` — behavioral patterns (sleep, food, routines)
   - `personal_facts` — relationships, locations, life events
   - `personality_traits` — inferred character traits
   - `communication_style` — message length, tone, emoji usage, formality
4. All inferences are grounded in actual conversation text — no guessing.

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Run pipeline (builds all artifacts)
```bash
# Full run (all segments + checkpoints — takes ~2-3 hours due to API calls)
python pipeline.py

# Quick demo run (first 50 segments, 20 checkpoints)
MAX_SEGS=50 MAX_CPS=20 python pipeline.py
```

### 4. Start chatbot
```bash
streamlit run app.py
```

Or with a custom artifacts directory:
```bash
ARTIFACTS_DIR=./artifacts streamlit run app.py
```

## Deployment (Hugging Face Spaces)

1. Create a new Space (Streamlit SDK)
2. Upload all files + `artifacts/` folder
3. Set `ANTHROPIC_API_KEY` in Space secrets
4. Set `ARTIFACTS_DIR=./artifacts` in Space secrets

## Stats

| Metric | Value |
|--------|-------|
| Total conversations | 11,001 |
| Total messages | 191,853 |
| Topic segments detected | ~18,025 |
| 100-msg checkpoints | 1,919 |
| TF-IDF vocab size | 5,000 |
| FAISS index type | IndexFlatIP |

## What We Don't Use

- No OpenAI / external embeddings model
- No LangChain / LlamaIndex (custom retrieval logic)
- Embeddings are TF-IDF (lightweight, no GPU needed)
- Claude API used only for summarization, persona extraction, and answer generation
