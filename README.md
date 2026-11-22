# Modified fork (2026-11-22)

This is a modified version of [666ghj/MiroFish](https://github.com/666ghj/MiroFish) (AGPL-3.0). Changes made on 2026-11-22:

- **LLM provider = Claude by default.** New `LLM_PROVIDER` setting: `anthropic` (default, official `anthropic` SDK) or `openai` (the previous OpenAI-compatible path: OpenAI/ChatGPT or, for example, a local Ollama server). The two services that built their own OpenAI client and the OASIS agents (camel `ModelPlatformType.ANTHROPIC`) follow the same setting.
- **Zep Cloud removed.** The knowledge graph is stored locally in SQLite (`backend/uploads/graph/graph.sqlite3`, standard library, FTS5 search). Entities and relations are extracted by the configured LLM using the project ontology. `zep-cloud` and `ZEP_API_KEY` are gone.
- **Google Fonts removed** from `frontend/index.html`; the UI uses the system font fallbacks already in the CSS.
- `docker-compose.yml` now builds this checkout (`build: .`) instead of pulling the upstream image.
- All Chinese text (comments, prompts, UI strings, Chinese README and Chinese-language images) was removed or translated to English.

Claude requests use `output_config.effort: "medium"`, no sampling parameters, and Anthropic's server-side refusal fallback (`fallbacks: "default"`, which may re-run a declined request on another Claude model).

**Resulting data flow:** with the default configuration the only external party that receives text (document chunks, ontology, extraction and simulation prompts) is the Anthropic API. The graph, search and all files stay on your machine.

**Models per stage.** MiroFish runs three LLM stages, and each one can use its own model. The `graph` stage (`LLM_MODEL_GRAPH`) generates the ontology and extracts entities from your uploaded documents. The `simulation` stage (`LLM_MODEL_SIMULATION`) covers agent profiles, the simulation config, the OASIS agents and the high-volume extraction of agent activity into the graph during a run. The `report` stage (`LLM_MODEL_REPORT`) covers the report agent, its chat and its tools. For each stage, its own variable wins; if it is unset, `LLM_MODEL_NAME` applies, so setting only `LLM_MODEL_NAME` gives you one model everywhere (for example `LLM_MODEL_NAME=claude-haiku-5-5` for the lowest cost). If neither is set, the provider's default applies: with `LLM_PROVIDER=anthropic` that is `claude-sonnet-5-5` for graph, `claude-haiku-5-5` for simulation and `claude-opus-5-5` for report. With `LLM_PROVIDER=openai` (OpenAI/ChatGPT or any OpenAI-compatible server such as a local Ollama) there are no per-stage defaults, so set `LLM_MODEL_NAME` and, optionally, the three stage variables. `LLM_BOOST_MODEL_NAME` still overrides the simulation model on the boost platform of a parallel simulation.

**Setup:** `cp .env.example .env`, then set `LLM_API_KEY` to an Anthropic API key. Nothing else is required. `.env.example` also has a commented block for OpenAI/ChatGPT.

**Not verified:**
- No live Claude API call was made; the Anthropic path was tested only against a mocked HTTP transport.
- The OASIS/camel simulation path (`backend/scripts/run_*_simulation.py`) was not executed (camel/oasis not installed); it uses camel's Anthropic backend, which calls Anthropic's OpenAI-compatible endpoint.
- Third-party dependencies were audited statically only (source read, nothing executed). Two automatic downloads remain; neither carries user text: the `Twitter/twhin-bert-base` model from `huggingface.co` (Twitter simulation only) and the tiktoken vocabulary from `openaipublic.blob.core.windows.net` (first token count). Both can be blocked with `HF_HUB_OFFLINE=1` and a preloaded `TIKTOKEN_CACHE_DIR`. Large binary packages (torch, nvidia-*) and the `mcp` package were not read.
- The API has no authentication and listens on all network interfaces by default (`FLASK_HOST`, `vite --host`, compose ports). This fork does not change that.
- All LLM prompts were translated from Chinese to English and were not run against a live model; model behaviour can differ from upstream.
- English is the only interface language. The other entries in `locales/languages.json` (es, fr, pt, ru, de) have no translation file, so they cannot be selected and the model is always told to answer in English.
- Graph search is lexical (BM25 over SQLite FTS5 trigrams), not semantic; results differ from Zep's hybrid search.
- The Docker image build (`docker compose up --build`) was not run.
- Extraction quality and cost on large documents were not measured (every chunk, and every simulated agent action written to the graph, is one or more LLM calls).

---

<div align="center">

<img src="./static/image/MiroFish_logo_compressed.jpeg" alt="MiroFish Logo" width="75%"/>

<a href="https://trendshift.io/repositories/16144" target="_blank"><img src="https://trendshift.io/api/badge/repositories/16144" alt="666ghj%2FMiroFish | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<em>A Simple and Universal Swarm Intelligence Engine, Predicting Anything</em>

<a href="https://www.shanda.com/" target="_blank"><img src="./static/image/shanda_logo.png" alt="666ghj%2FMiroFish | Shanda" height="40"/></a>

[![GitHub Stars](https://img.shields.io/github/stars/666ghj/MiroFish?style=flat-square&color=DAA520)](https://github.com/666ghj/MiroFish/stargazers)
[![GitHub Watchers](https://img.shields.io/github/watchers/666ghj/MiroFish?style=flat-square)](https://github.com/666ghj/MiroFish/watchers)
[![GitHub Forks](https://img.shields.io/github/forks/666ghj/MiroFish?style=flat-square)](https://github.com/666ghj/MiroFish/network)
[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/666ghj/MiroFish)

[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white)](http://discord.gg/ePf5aPaHnA)
[![X](https://img.shields.io/badge/X-Follow-000000?style=flat-square&logo=x&logoColor=white)](https://x.com/mirofish_ai)
[![Instagram](https://img.shields.io/badge/Instagram-Follow-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://www.instagram.com/mirofish_ai/)

</div>

## ⚡ Overview

**MiroFish** is a next-generation AI prediction engine powered by multi-agent technology. By extracting seed information from the real world (such as breaking news, policy drafts, or financial signals), it automatically constructs a high-fidelity parallel digital world. Within this space, thousands of intelligent agents with independent personalities, long-term memory, and behavioral logic freely interact and undergo social evolution. You can inject variables dynamically from a "God's-eye view" to precisely deduce future trajectories — **rehearse the future in a digital sandbox, and win decisions after countless simulations**.

> You only need to: Upload seed materials (data analysis reports or interesting novel stories) and describe your prediction requirements in natural language</br>
> MiroFish will return: A detailed prediction report and a deeply interactive high-fidelity digital world

### Our Vision

MiroFish is dedicated to creating a swarm intelligence mirror that maps reality. By capturing the collective emergence triggered by individual interactions, we break through the limitations of traditional prediction:

- **At the Macro Level**: We are a rehearsal laboratory for decision-makers, allowing policies and public relations to be tested at zero risk
- **At the Micro Level**: We are a creative sandbox for individual users — whether deducing novel endings or exploring imaginative scenarios, everything can be fun, playful, and accessible

From serious predictions to playful simulations, we let every "what if" see its outcome, making it possible to predict anything.

## 🌐 Live Demo

Welcome to visit our online demo environment and experience a prediction simulation on trending public opinion events we've prepared for you: [mirofish-live-demo](https://666ghj.github.io/mirofish-demo/)

## 🎬 Demo Videos

### 1. Wuhan University Public Opinion Simulation + MiroFish Project Introduction

[Watch the demo video: Wuhan University public opinion simulation (Bilibili)](https://www.bilibili.com/video/BV1VYBsBHEMY/)

The complete demo video of a prediction based on the BettaFish-generated "Wuhan University Public Opinion Report".

### 2. Dream of the Red Chamber Lost Ending Simulation

[Watch the demo video: Dream of the Red Chamber lost ending simulation (Bilibili)](https://www.bilibili.com/video/BV1cPk3BBExq)

MiroFish's deep prediction of the lost ending, based on hundreds of thousands of words from the first 80 chapters of "Dream of the Red Chamber".

> **Financial Prediction**, **Political News Prediction** and more examples coming soon...

## 🔄 Workflow

1. **Graph Building**: Seed extraction & Individual/collective memory injection & GraphRAG construction
2. **Environment Setup**: Entity relationship extraction & Persona generation & Agent configuration injection
3. **Simulation**: Dual-platform parallel simulation & Auto-parse prediction requirements & Dynamic temporal memory updates
4. **Report Generation**: ReportAgent with rich toolset for deep interaction with post-simulation environment
5. **Deep Interaction**: Chat with any agent in the simulated world & Interact with ReportAgent

## 🚀 Quick Start

### Option 1: Source Code Deployment (Recommended)

#### Prerequisites

| Tool | Version | Description | Check Installation |
|------|---------|-------------|-------------------|
| **Node.js** | 18+ | Frontend runtime, includes npm | `node -v` |
| **Python** | ≥3.11, ≤3.12 | Backend runtime | `python --version` |
| **uv** | Latest | Python package manager | `uv --version` |

#### 1. Configure Environment Variables

```bash
# Copy the example configuration file
cp .env.example .env

# Edit the .env file and fill in the required API keys
```

**Required Environment Variables:**

```env
# LLM configuration: anthropic (default) or openai (any OpenAI-compatible server, uses LLM_BASE_URL)
# High consumption, try simulations with fewer than 40 rounds first
LLM_PROVIDER=anthropic
LLM_API_KEY=your_anthropic_api_key
# Optional: one model for every stage (lowest cost). Without it, each stage uses its own default.
# LLM_MODEL_NAME=claude-haiku-5-5

# The knowledge graph is stored locally (SQLite); no graph service key is needed.
```

#### 2. Install Dependencies

```bash
# One-click installation of all dependencies (root + frontend + backend)
npm run setup:all
```

Or install step by step:

```bash
# Install Node dependencies (root + frontend)
npm run setup

# Install Python dependencies (backend, auto-creates virtual environment)
npm run setup:backend
```

#### 3. Start Services

```bash
# Start both frontend and backend (run from project root)
npm run dev
```

**Service URLs:**
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:5001`

**Start Individually:**

```bash
npm run backend   # Start backend only
npm run frontend  # Start frontend only
```

### Option 2: Docker Deployment

```bash
# 1. Configure environment variables (same as source deployment)
cp .env.example .env

# 2. Build the image from this checkout and start
docker compose up -d --build
```

Reads `.env` from root directory by default, maps ports `3000 (frontend) / 5001 (backend)`


## 📬 Join the Conversation

The MiroFish team is recruiting full-time/internship positions. If you're interested in multi-agent simulation and LLM applications, feel free to send your resume to: **mirofish@shanda.com**

## 📄 Acknowledgments

**MiroFish has received strategic support and incubation from Shanda Group!**

MiroFish's simulation engine is powered by **[OASIS (Open Agent Social Interaction Simulations)](https://github.com/camel-ai/oasis)**, We sincerely thank the CAMEL-AI team for their open-source contributions!

## 📈 Project Statistics

<a href="https://github.com/666ghj/MiroFish">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="static/image/star-history-dark.svg" />
   <source media="(prefers-color-scheme: light)" srcset="static/image/star-history-light.svg" />
   <img alt="666ghj/MiroFish Star History Chart" src="static/image/star-history-light.svg" />
 </picture>
</a>
