# TruthLens v2.0 — AI-Powered Fact Checker

> Semantic fact-checking powered by Groq (Llama 3.3) and real-time news data from NewsAPI.  
> Built with custom DSA implementations — no shortcuts, no libraries for the core logic.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3-8B5CF6?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What It Does

Enter any news claim — TruthLens fetches recent articles from across the web, runs semantic similarity analysis, and uses an LLM to determine whether the claim is **supported**, **contradicted**, or **unverified** by real journalism.

### Key Features
- 🔍 **Semantic Fact-Checking** — Groq Llama 3.3 analyzes claim vs. real articles
- 📰 **Live News Data** — pulls fresh articles from NewsAPI across 5 categories
- 📊 **Credibility Scoring** — per-article relationship tags (confirms / contradicts / partially confirms)
- 🚩 **Red Flag Detection** — highlights logical inconsistencies and unsupported claims
- 📈 **Visualization Tab** — Plotly charts for analysis history and score distribution
- 🗂️ **History & Patterns** — tracks all past verifications in session

---

## Data Structures (Implemented from Scratch)

This project uses **11 custom data structures** — no external DSA libraries:

| Structure | Use Case |
|---|---|
| **AVL Tree** | Self-balancing BST for sorted article indexing |
| **Red-Black Tree** | Efficient key-value store for cached results |
| **B-Tree / B+ Tree** | Multi-key search for article metadata |
| **Min/Max Heap** | Priority queue for top-scoring articles |
| **Huffman Tree** | Text compression for stored summaries |
| **Trie** | Prefix-based keyword indexing |
| **BK-Tree** | Fuzzy string matching for title similarity |
| **Skip List** | Probabilistic ordered search |
| **Union-Find** | Grouping related articles by topic cluster |
| **Persistent Array** | Immutable history snapshots |

---

## Tech Stack

- **Frontend** — Streamlit with custom CSS (dark theme)
- **LLM** — Groq API (Llama 3.3 70B)
- **News Data** — NewsAPI.org
- **Charts** — Plotly
- **Language** — Python 3.9+

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/truthlens.git
cd truthlens
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

You'll need:
- **Groq API key** — free at [console.groq.com](https://console.groq.com)
- **NewsAPI key** — free at [newsapi.org](https://newsapi.org)

### 4. Run the app

```bash
streamlit run app.py
```

---

## Project Structure

```
truthlens/
├── app.py              # Main application (all logic + UI)
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── .gitignore
└── README.md
```

---

## Screenshots

> _Add a screenshot or demo GIF here after deploying_

---

## Live Demo

> _Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) and add the link here — it's free!_

---

## Author

**Pankaj** — SY B.Tech CSE (AIML), Vishwakarma Institute of Technology, Pune

---

## License

MIT — free to use and modify.
