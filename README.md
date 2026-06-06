# AI in Learning Design Advisor

**For learning practitioners: evidence-based guidance on friction in learning and the role of AI.**

A RAG-based conversational advisor that helps instructional designers, L&D practitioners, facilitators, teachers, and faculty reason clearly about their design decisions -- and where AI belongs in them.

Built as Assignment 1 for *Building Agentic AI Applications for Beginners*, Codecademy, 2026.

🌐 **Live app:** [ai-learning-design-advisor.streamlit.app](https://ai-learning-design-advisor.streamlit.app)

---

## Demo

📹 [Watch the demo](https://drive.google.com/file/d/1DA9PYYuDUWZo_Z1IhWqeJUGBbOStjy_P/view?usp=sharing)

Two practitioner scenarios: a manager training redesign with a sensitive coaching component, and a faculty instructor thinking proactively about AI use in student reflection papers.

---

## What It Does

The advisor takes a practitioner's question or learning problem and responds with:

- A grounded diagnosis of what kind of challenge it actually is
- A clear recommendation on how and where AI fits
- Evidence drawn from learning science research and practitioner frameworks

The diagnostic layer is built on the **Three Friction Types Framework** (Chapman, 2026), which distinguishes between overhead that should be removed, complexity that should be scaffolded, and cognitive work that should be protected. The agent uses this framework to reason about each question, not to label it, but to shape the recommendation.

---

## Architecture

```
User Input
    │
    ▼
Retriever (FAISS vector store)
    │
    ▼
Retrieved Chunks (top 5)
    │
    ├── Knowledge Base (19 documents: learning science research)
    └── Three Friction Types Framework (Chapman, 2026)
    │
    ▼
Claude Haiku (claude-haiku-4-5-20251001)
    │
    ▼
Practitioner-facing response (1-2 paragraphs + 2-4 recommendations)
```

The system uses a `stuff` chain -- retrieved chunks are passed directly into the prompt context window alongside the question. No summarization or map-reduce step.

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | Claude Haiku (Anthropic) | Fast, affordable, strong instruction-following |
| Embeddings | SentenceTransformer `all-MiniLM-L6-v2` | No API cost, high quality for semantic retrieval |
| Vector store | FAISS | Lightweight, runs locally without a managed service |
| Framework | LangChain + langchain-classic | Stable RetrievalQA chain support |
| UI | Streamlit | Fast to deploy, practitioner-friendly interface |
| Hosting | Streamlit Cloud | Free tier, deploys directly from GitHub |

---

## Knowledge Base

19 documents covering learning science research and practitioner frameworks, including:

- Pearson Learning Design Principles series (attention/cognitive load, authentic learning, developing understanding, digital learning, motivation/mindset, objective design)
- Mayer's 12 Principles of Multimedia Learning
- Khosravi et al. (2026), *Building AI Companions that Prioritise Learning over Performance*
- Hardman newsletters: *Hidden Cost of AI-Generated Feedback*, *From AI Tutors to AI Study Mates*, *Beyond the Hype: 18 Studies*
- AACE (2026), *Generative AI for Instructional Design*
- CESE Cognitive Load Theory (research and practice editions)
- US Department of Education, *AI and the Future of Teaching and Learning* (2023)
- Illinois Best Practices for Teaching and Learning with GenAI
- Chapman, N. (2026), *Three Friction Types Framework* [proprietary]

The knowledge base is intentionally constrained. This tool is not a general AI assistant -- it is grounded in a specific body of learning science literature, practitioner frameworks, and a proprietary diagnostic framework. That constraint is what makes the recommendations defensible.

---

## Running Locally

**Prerequisites:** Python 3.9+, an Anthropic API key.

```bash
git clone https://github.com/nhchapman05/ai-learning-design-advisor.git
cd ai-learning-design-advisor
pip install -r requirements.txt
```

Add your Anthropic API key to `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-key-here"
```

Then run:

```bash
streamlit run app.py
```

The knowledge base loads and indexes on first run. Subsequent loads use Streamlit's cache.

---

## Known Limitations

- **Retrieval breadth:** The current setup surfaces a small number of unique sources per query. Expanding `k` or switching to a managed vector store would improve coverage.
- **Single-word inputs:** Queries with minimal context produce weak retrieval results. The agent is designed for practitioner-level questions, not keyword lookups.
- **Multi-turn context:** Conversation history is displayed within a session but is not consistently passed into the retrieval chain. The model may or may not draw on earlier turns depending on how the user phrases follow-up questions. A future build would explicitly pass conversation history into the prompt.
- **Response length:** The system prompt constrains responses to 1-2 paragraphs and 2-4 recommendations. Occasional drift toward longer responses is a known issue under active refinement.

---

## Future Enhancements

- **Multi-agent architecture:** Separate Diagnostic, Reasoning, and Output agents for more structured and auditable recommendations.
- **Governance Creator:** A second mode or sub-agent that helps organizations build AI governance language for their learning programs, drawing on UNESCO, NIST AI RMF, and EU AI Act.
- **Expanded knowledge base:** Additional peer-reviewed sources, practitioner case studies, and institutional AI policies.
- **Persistent memory:** Cross-session conversation history so the agent can build context with a returning practitioner over time.
- **Better retrieval:** Hybrid search (keyword + semantic) and re-ranking to surface more diverse sources per query.

---

## About

Built by **Natasha Chapman, PhD** -- Learning Experience Designer with 20 years of practice across higher education, leadership development, and corporate L&D. Founder of Lit Learning LLC.

This project is the first working layer of a larger product: a practitioner-facing diagnostic tool that helps learning teams reason about where AI belongs in their design process -- and where it doesn't.
