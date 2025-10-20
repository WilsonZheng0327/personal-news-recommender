# News Recommender System - Complete Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Tech Stack Summary](#tech-stack-summary)
6. [Deployment Architecture](#deployment-architecture)
7. [RAG System Deep Dive](#rag-system-deep-dive)

---

## System Overview

### What We're Building
A personalized news recommendation platform that:
- Scrapes articles from multiple sources
- Classifies articles by topic using a fine-tuned model
- Generates personalized recommendations using collaborative filtering + content-based approaches
- **Provides a conversational RAG interface for natural language Q&A about news**
- Serves recommendations through a web interface
- Continuously learns from user interactions

### Core Components
1. **Data Collection Layer** - Scrape and ingest articles
2. **ML Training Pipeline** - Fine-tune topic classification model
3. **Processing Pipeline** - Classify and embed articles
4. **User Tracking System** - Capture interactions
5. **Recommendation Engine** - Generate personalized feeds
6. **RAG System (Conversational Assistant)** - Natural language Q&A over news articles
7. **API & Frontend** - Serve to users
8. **Infrastructure & Monitoring** - Deploy and observe

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SOURCES                            │
│  RSS Feeds │ News APIs │ Reddit │ Twitter │ Other Sources           │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      1. DATA COLLECTION LAYER                        │
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ RSS Scrapers │───▶│ Article      │───▶│ PostgreSQL   │          │
│  │ (feedparser) │    │ Extractors   │    │ (Raw Data)   │          │
│  └──────────────┘    │ (newspaper3k)│    └──────────────┘          │
│                      └──────────────┘                                │
│  Celery Beat (Scheduler) + Redis (Message Broker)                   │
│  Runs every hour: Fetch new articles, deduplicate, store            │
└────────────┬────────────────────────────────────────────────────────┘
             │
             │ New articles stored
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    2. ML TRAINING PIPELINE                           │
│                       (One-time + Periodic)                          │
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ AG News      │───▶│ HuggingFace  │───▶│ Fine-tuned   │          │
│  │ Dataset      │    │ Transformers │    │ DistilBERT   │          │
│  │ (120K texts) │    │ + PyTorch    │    │ Classifier   │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                   │                  │
│  Training on: Google Colab (Free GPU) or AWS SageMaker             │
│  Tracking: WandB (experiments) + MLflow (model registry)            │
│  Duration: ~2-3 hours                                               │
│                                                   │                  │
│                                                   │ Model artifact   │
│                                                   ▼                  │
│                                          ┌──────────────┐            │
│                                          │ Model Store  │            │
│                                          │ (S3/Local)   │            │
│                                          └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
             │
             │ Model deployed
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    3. ARTICLE PROCESSING PIPELINE                    │
│                         (Background Workers)                         │
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ New Article  │───▶│ Topic        │───▶│ Update       │          │
│  │ from DB      │    │ Classifier   │    │ Article with │          │
│  │              │    │ (Fine-tuned) │    │ Topic Label  │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                   │                  │
│                           ┌───────────────────────┘                 │
│                           ▼                                          │
│                  ┌──────────────┐                                    │
│                  │ Embedding    │                                    │
│                  │ Generator    │                                    │
│                  │ (Sentence-   │                                    │
│                  │ Transformers)│                                    │
│                  └──────┬───────┘                                    │
│                         │                                            │
│                         ▼                                            │
│                  ┌──────────────┐    ┌──────────────┐              │
│                  │ FAISS Index  │───▶│ Vector DB    │              │
│                  │ (384-dim     │    │ Storage      │              │
│                  │  vectors)    │    │              │              │
│                  └──────────────┘    └──────────────┘              │
│                                                                       │
│  Celery Workers: Process unclassified articles every 10 min         │
│  Each article gets: topic label + confidence + embedding vector     │
└─────────────────────────────────────────────────────────────────────┘
             │
             │ Processed articles ready
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    4. USER INTERACTION TRACKING                      │
│                                                                       │
│  User Action (Click/Read/Like/Skip)                                 │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ FastAPI      │───▶│ PostgreSQL   │───▶│ User Profile │          │
│  │ Endpoint     │    │ Interaction  │    │ Builder      │          │
│  │              │    │ Table        │    │              │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                   │                  │
│                           ┌───────────────────────┘                 │
│                           ▼                                          │
│                  ┌──────────────┐                                    │
│                  │ User Profile │                                    │
│                  │ Cache        │                                    │
│                  │ (Redis)      │                                    │
│                  │              │                                    │
│                  │ - Topic prefs│                                    │
│                  │ - Avg embed  │                                    │
│                  │ - Read history                                    │
│                  └──────────────┘                                    │
│                                                                       │
│  Updates: Real-time on user action                                  │
│  Profile rebuild: Async background job every 1 hour                 │
└─────────────────────────────────────────────────────────────────────┘
             │
             │ User profiles updated
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    5. RECOMMENDATION ENGINE                          │
│                      (Multiple Models Combined)                      │
│                                                                       │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              Model 1: Content-Based Filtering         │           │
│  │                                                        │           │
│  │  User Profile Vector (384-dim)                        │           │
│  │         │                                              │           │
│  │         ▼                                              │           │
│  │  ┌──────────────┐         ┌──────────────┐           │           │
│  │  │ FAISS        │────────▶│ Top-K        │           │           │
│  │  │ Similarity   │  cosine │ Similar      │           │           │
│  │  │ Search       │  distance│ Articles    │           │           │
│  │  └──────────────┘         └──────┬───────┘           │           │
│  │                                   │                   │           │
│  │                                   │ Score 1           │           │
│  └───────────────────────────────────┼───────────────────┘           │
│                                      │                               │
│  ┌──────────────────────────────────┼───────────────────┐           │
│  │      Model 2: Collaborative Filtering (ALS)          │           │
│  │                                   │                   │           │
│  │  User-Item Interaction Matrix     │                   │           │
│  │  (Sparse matrix of clicks)        │                   │           │
│  │         │                          │                   │           │
│  │         ▼                          │                   │           │
│  │  ┌──────────────┐                 │                   │           │
│  │  │ Implicit ALS │                 │                   │           │
│  │  │ Model        │                 │                   │           │
│  │  │ (Matrix      │                 │                   │           │
│  │  │ Factorization)│                │                   │           │
│  │  └──────┬───────┘                 │                   │           │
│  │         │                          │                   │           │
│  │         │ Predicts user-item      │                   │           │
│  │         │ affinity scores          │                   │           │
│  │         │                          │                   │           │
│  │         ▼                          │                   │           │
│  │  ┌──────────────┐                 │                   │           │
│  │  │ Recommend    │─────────────────┤ Score 2           │           │
│  │  │ Articles for │                 │                   │           │
│  │  │ User         │                 │                   │           │
│  │  └──────────────┘                 │                   │           │
│  │                                   │                   │           │
│  │  Retrained: Daily on new interaction data            │           │
│  └──────────────────────────────────┼───────────────────┘           │
│                                      │                               │
│  ┌──────────────────────────────────┼───────────────────┐           │
│  │         Model 3: Topic-Based Filtering               │           │
│  │                                   │                   │           │
│  │  User Topic Preferences           │                   │           │
│  │  (world: 0.1, sports: 0.5, ...)  │                   │           │
│  │         │                          │                   │           │
│  │         ▼                          │                   │           │
│  │  ┌──────────────┐                 │                   │           │
│  │  │ Match with   │─────────────────┤ Score 3           │           │
│  │  │ Article      │                 │                   │           │
│  │  │ Topics       │                 │                   │           │
│  │  └──────────────┘                 │                   │           │
│  └──────────────────────────────────┼───────────────────┘           │
│                                      │                               │
│  ┌──────────────────────────────────┼───────────────────┐           │
│  │         Model 4: Popularity & Recency                │           │
│  │                                   │                   │           │
│  │  - Trending articles (many clicks)│                   │           │
│  │  - Recent articles (published today)                 │           │
│  │  - Diversity boost                │                   │           │
│  │         │                          │                   │           │
│  │         ▼                          │                   │           │
│  │  ┌──────────────┐                 │                   │           │
│  │  │ Trending     │─────────────────┤ Score 4           │           │
│  │  │ Score        │                 │                   │           │
│  │  └──────────────┘                 │                   │           │
│  └──────────────────────────────────┼───────────────────┘           │
│                                      │                               │
│                      All scores      │                               │
│                           │          │                               │
│                           ▼          ▼                               │
│                  ┌──────────────────────┐                            │
│                  │  Ensemble Combiner   │                            │
│                  │                      │                            │
│                  │  Final Score =       │                            │
│                  │    0.35 * Score1 +   │                            │
│                  │    0.35 * Score2 +   │                            │
│                  │    0.20 * Score3 +   │                            │
│                  │    0.10 * Score4     │                            │
│                  └──────────┬───────────┘                            │
│                             │                                        │
│                             ▼                                        │
│                  ┌──────────────────────┐                            │
│                  │  Ranking & Filtering │                            │
│                  │                      │                            │
│                  │  - Remove already read                            │
│                  │  - Diversity (topics)│                            │
│                  │  - Apply business    │                            │
│                  │    rules             │                            │
│                  └──────────┬───────────┘                            │
│                             │                                        │
│                             ▼                                        │
│                  ┌──────────────────────┐                            │
│                  │  Cache in Redis      │                            │
│                  │  (30 min TTL)        │                            │
│                  └──────────────────────┘                            │
│                                                                       │
│  Pre-computed: Every 30 min for active users                        │
│  On-demand: For new users or cache miss                             │
└─────────────────────────────────────────────────────────────────────┘
             │
             │ Recommendations ready
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  6. RAG SYSTEM (Conversational Assistant)            │
│                      Natural Language Q&A over News                  │
│                                                                       │
│  User Query: "What's the latest on AI regulation in the EU?"        │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              Query Processing                         │           │
│  │                                                        │           │
│  │  ┌──────────────┐         ┌──────────────┐           │           │
│  │  │ Conversation │────────▶│ Query        │           │           │
│  │  │ Memory       │ context │ Understanding│           │           │
│  │  │ (Redis)      │         │              │           │           │
│  │  └──────────────┘         └──────┬───────┘           │           │
│  │                                   │                   │           │
│  │                                   │ Enhanced query    │           │
│  └───────────────────────────────────┼───────────────────┘           │
│                                      │                               │
│  ┌──────────────────────────────────┼───────────────────┐           │
│  │              Retrieval Stage                          │           │
│  │                                   │                   │           │
│  │                                   ▼                   │           │
│  │                          ┌──────────────┐            │           │
│  │                          │ Embed Query  │            │           │
│  │                          │ (Sentence-   │            │           │
│  │                          │ Transformers)│            │           │
│  │                          └──────┬───────┘            │           │
│  │                                 │                    │           │
│  │                                 ▼                    │           │
│  │  ┌──────────────┐      ┌──────────────┐            │           │
│  │  │ FAISS Index  │─────▶│ Top-K        │            │           │
│  │  │ Similarity   │cosine│ Articles     │            │           │
│  │  │ Search       │search│ (K=5-10)     │            │           │
│  │  └──────────────┘      └──────┬───────┘            │           │
│  │                               │                     │           │
│  │                               │ Candidate articles  │           │
│  │                               ▼                     │           │
│  │                      ┌──────────────┐              │           │
│  │                      │ Re-ranker    │              │           │
│  │                      │ (Optional)   │              │           │
│  │                      │ - Recency    │              │           │
│  │                      │ - Relevance  │              │           │
│  │                      │ - User prefs │              │           │
│  │                      └──────┬───────┘              │           │
│  │                             │                      │           │
│  │                             │ Top-5 articles       │           │
│  └─────────────────────────────┼──────────────────────┘           │
│                                │                                   │
│  ┌─────────────────────────────┼──────────────────────┐           │
│  │           Generation Stage                          │           │
│  │                             │                       │           │
│  │                             ▼                       │           │
│  │                   ┌──────────────────┐             │           │
│  │                   │ Context Builder  │             │           │
│  │                   │                  │             │           │
│  │                   │ Combines:        │             │           │
│  │                   │ - User query     │             │           │
│  │                   │ - Retrieved docs │             │           │
│  │                   │ - Conversation   │             │           │
│  │                   │   history        │             │           │
│  │                   │ - System prompt  │             │           │
│  │                   └────────┬─────────┘             │           │
│  │                            │                       │           │
│  │                            │ Full context          │           │
│  │                            ▼                       │           │
│  │                   ┌──────────────────┐             │           │
│  │                   │ LLM              │             │           │
│  │                   │ (GPT-4 / Llama)  │             │           │
│  │                   │                  │             │           │
│  │                   │ Generate answer  │             │           │
│  │                   │ with citations   │             │           │
│  │                   └────────┬─────────┘             │           │
│  │                            │                       │           │
│  │                            │ Generated response    │           │
│  └────────────────────────────┼───────────────────────┘           │
│                               │                                   │
│                               ▼                                   │
│                      ┌──────────────────┐                         │
│                      │ Post-Processing  │                         │
│                      │                  │                         │
│                      │ - Format answer  │                         │
│                      │ - Add citations  │                         │
│                      │ - Track sources  │                         │
│                      │ - Store in memory│                         │
│                      └────────┬─────────┘                         │
│                               │                                   │
│                               ▼                                   │
│                      Response with citations                      │
│                                                                       │
│  Tech Stack:                                                        │
│  - LLM: OpenAI API (GPT-4), Ollama (local), or Together.ai        │
│  - Orchestration: LangChain or custom                              │
│  - Memory: Redis (short-term) + PostgreSQL (long-term)            │
│  - Evaluation: RAGAS metrics, human feedback                       │
│                                                                       │
│  Use Cases Handled:                                                 │
│  ✓ Q&A about current events                                        │
│  ✓ Article summarization                                           │
│  ✓ Topic explanation                                               │
│  ✓ Multi-article synthesis                                         │
│  ✓ Context/background research                                     │
│  ✓ Comparison across sources                                       │
└─────────────────────────────────────────────────────────────────────┘
             │
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      7. API & FRONTEND LAYER                         │
│                                                                       │
│  ┌──────────────┐                                                    │
│  │   React      │                                                    │
│  │   Frontend   │                                                    │
│  │              │                                                    │
│  │  - Feed view │                                                    │
│  │  - Article   │                                                    │
│  │    reader    │                                                    │
│  │  - Profile   │                                                    │
│  │    settings  │                                                    │
│  └──────┬───────┘                                                    │
│         │                                                            │
│         │ HTTP/REST                                                  │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────┐                       │
│  │         FastAPI Backend                   │                       │
│  │                                           │                       │
│  │  Endpoints:                               │                       │
│  │  - GET  /api/feed                         │                       │
│  │  - POST /api/interaction                  │                       │
│  │  - GET  /api/article/{id}                 │                       │
│  │  - GET  /api/profile                      │                       │
│  │  - GET  /api/topics                       │                       │
│  │  - GET  /api/similar/{article_id}         │                       │
│  │  - GET  /health                           │                       │
│  │                                           │                       │
│  │  Middleware:                              │                       │
│  │  - Authentication (JWT)                   │                       │
│  │  - Rate limiting                          │                       │
│  │  - CORS                                   │                       │
│  │  - Logging                                │                       │
│  └──────┬────────────────────────────────────┘                       │
│         │                                                            │
│         │ Queries/Updates                                            │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ PostgreSQL   │    │ Redis Cache  │    │ FAISS Index  │          │
│  │              │    │              │    │              │          │
│  │ - Users      │    │ - Recs       │    │ - Vectors    │          │
│  │ - Articles   │    │ - Profiles   │    │              │          │
│  │ - Interactions    │ - Sessions   │    │              │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
             │
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  8. INFRASTRUCTURE & MONITORING                      │
│                                                                       │
│  ┌──────────────────────────────────────────┐                       │
│  │            AWS Infrastructure             │                       │
│  │                                           │                       │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────┐│                       │
│  │  │ EC2      │  │ RDS      │  │ S3      ││                       │
│  │  │ (Backend)│  │ (Postgres│  │ (Models,││                       │
│  │  │ (Workers)│  │   DB)    │  │ Logs)   ││                       │
│  │  └──────────┘  └──────────┘  └─────────┘│                       │
│  │                                           │                       │
│  │  ┌──────────┐  ┌──────────┐             │                       │
│  │  │ElastiCache  │ CloudWatch               │                       │
│  │  │ (Redis)  │  │(Monitoring│             │                       │
│  │  └──────────┘  └──────────┘             │                       │
│  │                                           │                       │
│  │  ┌──────────────────────────┐            │                       │
│  │  │ Application Load Balancer│            │                       │
│  │  └──────────────────────────┘            │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                       │
│  ┌──────────────────────────────────────────┐                       │
│  │         Monitoring & Logging              │                       │
│  │                                           │                       │
│  │  - CloudWatch: System metrics             │                       │
│  │  - Custom dashboards: Business metrics    │                       │
│  │  - Prometheus + Grafana: Advanced         │                       │
│  │  - ELK Stack: Log aggregation             │                       │
│  │  - Sentry: Error tracking                 │                       │
│  │                                           │                       │
│  │  Metrics tracked:                         │                       │
│  │  - API latency (p50, p95, p99)            │                       │
│  │  - Recommendation quality (CTR, diversity)│                       │
│  │  - Model performance (accuracy drift)     │                       │
│  │  - System health (CPU, memory, errors)    │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                       │
│  ┌──────────────────────────────────────────┐                       │
│  │            CI/CD Pipeline                 │                       │
│  │                                           │                       │
│  │  GitHub Actions:                          │                       │
│  │  - Run tests on PR                        │                       │
│  │  - Build Docker images                    │                       │
│  │  - Deploy to staging                      │                       │
│  │  - Manual approval for prod              │                       │
│  │  - Deploy to production                   │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Collection Layer

**Purpose**: Continuously gather news articles from various sources

**Tech Stack**:
- **Scrapers**: `feedparser` (RSS), `newspaper3k` (full text extraction), `requests`, `beautifulsoup4`
- **Task Queue**: `Celery` + `Redis` (message broker)
- **Scheduler**: `Celery Beat` (cron-like scheduling)
- **Storage**: `PostgreSQL` (structured data), `SQLAlchemy` (ORM)

**How It Works**:
1. **Celery Beat** triggers scraping tasks every hour
2. **Scrapers** fetch articles from RSS feeds and APIs
3. **Extractors** download full article content and parse metadata
4. **Deduplication** checks if article URL already exists in database
5. **Storage** saves new articles to PostgreSQL with status="unprocessed"

**Interactions**:
- → Writes to PostgreSQL (`articles` table)
- → Triggers processing pipeline (via database status flag)
- ← Reads from external RSS feeds/APIs

**Key Considerations**:
- Rate limiting to avoid being blocked
- Error handling for failed scrapes
- Incremental updates (only fetch new articles)
- Source diversity (multiple news outlets)

---

### 2. ML Training Pipeline

**Purpose**: Fine-tune a transformer model to classify articles by topic

**Tech Stack**:
- **Framework**: `PyTorch` + `HuggingFace Transformers`
- **Dataset**: AG News (120K labeled articles) via `HuggingFace datasets`
- **Compute**: Google Colab (free GPU) or AWS SageMaker
- **Tracking**: `WandB` (experiment tracking), `MLflow` (model registry)
- **Optimization**: `accelerate` (training speed)

**How It Works**:
1. **Data Preparation**: Load AG News dataset (4 categories: World, Sports, Business, Tech)
2. **Model Selection**: Start with `distilbert-base-uncased` (fast, good accuracy)
3. **Fine-tuning**: Train for 3 epochs with learning rate 2e-5
4. **Evaluation**: Measure accuracy, F1 score, confusion matrix
5. **Model Registry**: Save best model to MLflow/S3
6. **Deployment**: Download model to production server

**Interactions**:
- ← Reads AG News dataset (external)
- → Saves model artifacts to S3 or local storage
- → Logs experiments to WandB
- → Registers model in MLflow

**Key Considerations**:
- One-time training initially, then periodic retraining (monthly/quarterly)
- Hyperparameter tuning (learning rate, batch size, epochs)
- Model versioning (track performance over time)
- A/B testing new model versions vs. old

**Training Duration**: ~2-3 hours on free Colab GPU

---

### 3. Article Processing Pipeline

**Purpose**: Classify and embed each article for downstream recommendations

**Tech Stack**:
- **Classification**: Fine-tuned DistilBERT (from step 2)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Search**: `FAISS` (Facebook AI Similarity Search)
- **Task Queue**: `Celery` workers
- **Model Serving**: `transformers` pipelines

**How It Works**:
1. **Celery Worker** queries for articles with `topic=NULL` (unprocessed)
2. **Topic Classifier** predicts topic (world/sports/business/tech) + confidence score
3. **Update Database** with topic label
4. **Embedding Generator** creates 384-dim vector from article text
5. **FAISS Index** adds vector for similarity search
6. **Store Mapping** between article ID and FAISS index position

**Interactions**:
- ← Reads unprocessed articles from PostgreSQL
- → Updates PostgreSQL with topic labels
- → Writes embeddings to FAISS index
- ← Loads fine-tuned model from storage
- ← Loads sentence-transformer model (pre-trained)

**Key Considerations**:
- Batch processing (100 articles at a time) for efficiency
- GPU usage for faster inference (optional)
- Error handling (skip corrupted articles)
- Index persistence (save FAISS to disk periodically)

**Processing Speed**: ~10-20 articles/second on CPU

---

### 4. User Interaction Tracking

**Purpose**: Capture user behavior and build preference profiles

**Tech Stack**:
- **Database**: `PostgreSQL` (interactions table)
- **Cache**: `Redis` (user profiles)
- **API**: `FastAPI` endpoints
- **Analytics**: `pandas` for aggregations

**How It Works**:
1. **Frontend** sends interaction event (click/read/like/skip)
2. **FastAPI** validates and logs to `interactions` table
3. **Background Job** (Celery) rebuilds user profiles periodically
4. **Profile Builder** aggregates:
   - Topic preferences (% of reads per topic)
   - Average embedding vector (mean of liked articles)
   - Reading patterns (time of day, article length)
5. **Cache** stores profiles in Redis for fast access

**Interactions**:
- ← Receives events from FastAPI
- → Writes to PostgreSQL (`interactions` table)
- → Updates Redis cache (user profiles)
- ← Reads article embeddings from FAISS
- → Provides profiles to recommendation engine

**Key Considerations**:
- Real-time updates for immediate feedback
- Privacy (GDPR compliance, anonymization)
- Cold start problem (new users with no history)
- Profile decay (older interactions matter less)

---

### 5. Recommendation Engine

**Purpose**: Generate personalized article rankings combining multiple signals

**Tech Stack**:
- **Collaborative Filtering**: `implicit` library (ALS algorithm)
- **Content-Based**: FAISS similarity search
- **ML Models**: `scikit-learn` (additional models)
- **Sparse Matrices**: `scipy.sparse`
- **Caching**: `Redis`

**How It Works**:

#### Model 1: Content-Based Filtering
- Get user's profile embedding (average of liked articles)
- Use FAISS to find top-K similar articles
- Score based on cosine similarity

#### Model 2: Collaborative Filtering
- Build user-item interaction matrix (sparse)
- Train ALS model (matrix factorization)
- Predict affinity scores for unseen articles
- Retrain daily on new interactions

#### Model 3: Topic-Based
- Match user's topic preferences (sports: 0.5, tech: 0.3, ...)
- Score articles by topic alignment

#### Model 4: Popularity & Recency
- Trending articles (most clicks in last 24h)
- Recent articles (published today)
- Diversity boost (ensure variety)

#### Ensemble Combiner
```
Final Score = 0.35 * ContentBased + 
              0.35 * Collaborative + 
              0.20 * TopicBased + 
              0.10 * Popularity
```

#### Post-Processing
- Remove already-read articles
- Ensure topic diversity
- Apply business rules (e.g., min 2 articles from each category)

**Interactions**:
- ← Reads user profiles from Redis
- ← Reads embeddings from FAISS
- ← Reads interactions from PostgreSQL
- → Caches recommendations in Redis (30 min TTL)
- → Logs recommendation quality metrics

**Key Considerations**:
- Pre-computation for active users (background job)
- On-demand for new users or cache miss
- A/B testing different ensemble weights
- Explainability ("Why this article?")

**Generation Time**: 50-200ms per user (cached: <10ms)

---

### 6. RAG System (Conversational News Assistant)

**Purpose**: Enable natural language Q&A over the news article database using Retrieval Augmented Generation

**Tech Stack**:
```yaml
LLM Options:
  - OpenAI API: GPT-4-turbo or GPT-3.5-turbo (easiest, best quality)
  - Local Models: Ollama + Llama 3 / Mistral (cost-effective, private)
  - Hosted Inference: Together.ai, Anyscale, Replicate (scalable)

RAG Framework:
  - LangChain: High-level RAG orchestration (optional)
  - Custom Pipeline: Direct control, shows deeper understanding

Vector Search:
  - FAISS: Already implemented for embeddings
  - sentence-transformers: Query embedding

Conversation Management:
  - Redis: Short-term conversation history (last 10 messages)
  - PostgreSQL: Long-term chat logs, analytics

Prompt Engineering:
  - System prompts: Define assistant behavior
  - Few-shot examples: Improve response quality
  - Template management: Different prompts for different query types

Evaluation:
  - RAGAS: RAG-specific metrics (faithfulness, relevance)
  - Human feedback: Thumbs up/down on responses
  - Citation accuracy: Do sources support claims?
```

**How It Works**:

#### Stage 1: Query Processing
1. **User submits question** via chat interface
2. **Conversation Memory** retrieves recent context (last 3-5 turns)
3. **Query Understanding** (optional):
   - Detect query intent (summarize vs. explain vs. compare)
   - Resolve pronouns ("this" → current article, "they" → entity from context)
   - Query rewriting for better retrieval

#### Stage 2: Retrieval
1. **Embed Query** using sentence-transformers (same model as articles)
2. **FAISS Search** finds top-K semantically similar articles (K=5-10)
3. **Filtering**:
   - Remove articles user already discussed
   - Filter by recency (optional: prioritize recent articles)
   - Filter by user preferences (optional: topics they care about)
4. **Re-ranking** (optional but recommended):
   - Combine semantic similarity with:
     - Recency score (newer = better)
     - Popularity score (more clicks = better)
     - Topic alignment (user preferences)
   - Final top-5 articles selected

#### Stage 3: Context Building
1. **Gather Context**:
   - User query
   - Top-5 retrieved articles (title + full text)
   - Conversation history (last 3 messages)
   - User profile (topics of interest, reading level)
2. **Build Prompt**:
```
System: You are a news assistant. Answer questions based on provided articles.
Always cite sources. Be concise but comprehensive.

Articles:
[1] Title: "EU Proposes New AI Regulations"
    Source: TechCrunch, 2025-01-15
    Content: ...

[2] Title: "Tech Companies React to AI Rules"
    Source: Reuters, 2025-01-16
    Content: ...

Conversation History:
User: What's happening with AI in Europe?
Assistant: The EU recently proposed new AI regulations...

Current Question: What are the main concerns from tech companies?
```

3. **Token Management**:
   - Check context length doesn't exceed model limit (e.g., 8K tokens)
   - Truncate articles if needed (keep title + first paragraphs)
   - Prioritize most relevant articles

#### Stage 4: Generation
1. **LLM Call** with constructed prompt
2. **Generation Parameters**:
   - Temperature: 0.3 (focused, factual responses)
   - Max tokens: 500-800 (concise but complete)
   - Top-p: 0.9 (balanced creativity)
3. **Streaming** (optional): Stream response token-by-token for better UX

#### Stage 5: Post-Processing
1. **Extract Citations**: Identify which articles were used
2. **Format Response**:
   - Clean up formatting
   - Add source links
   - Highlight citations
3. **Store in Memory**:
   - Save conversation turn to Redis
   - Log to PostgreSQL for analytics
4. **Track Sources**: Record which articles influenced answer

**Interactions**:
- ← Receives queries from FastAPI chat endpoint
- ↔ Queries FAISS for similar articles
- ← Reads article content from PostgreSQL
- ← Loads conversation history from Redis
- → Calls LLM API (OpenAI, Ollama, etc.)
- → Stores conversation in Redis + PostgreSQL
- → Returns answer with citations to frontend

**Use Cases This Handles**:

1. **Current Events Q&A**:
   - "What's the latest on the US election?"
   - "Tell me about recent tech layoffs"

2. **Article Summarization**:
   - "Summarize the top 3 articles about climate change"
   - "Give me a brief summary of this article: [URL]"

3. **Topic Explanation**:
   - "What is quantitative easing?"
   - "Explain the background of the Ukraine conflict"

4. **Context Research**:
   - "What events led to this situation?"
   - "Has this happened before?"

5. **Multi-Source Synthesis**:
   - "How are different sources covering this story?"
   - "What's the consensus on this issue?"

6. **Comparison**:
   - "Compare CNN and Fox News coverage of [event]"
   - "What are the different perspectives on this?"

**Key Considerations**:
- **Hallucination Prevention**: Ground all responses in retrieved articles
- **Citation Accuracy**: Every claim should reference a source
- **Context Window**: Manage token limits carefully
- **Latency**: Target <3 seconds for full response (<1s with streaming)
- **Cost**: LLM API calls are expensive (cache common queries)
- **Evaluation**: Continuously measure response quality

**Performance Targets**:
- Retrieval: <100ms (FAISS is fast)
- LLM generation: 1-3 seconds (depends on model)
- Total response time: <4 seconds
- Citation accuracy: >95% (claims match sources)
- User satisfaction: >80% thumbs up

---

### 7. API & Frontend Layer

**Purpose**: Serve recommendations and handle user interactions

**Tech Stack**:
- **Backend**: `FastAPI` (async Python web framework)
- **Frontend**: `React` + `Axios` (or vanilla JS + Tailwind)
- **Authentication**: `JWT` tokens
- **Database Driver**: `SQLAlchemy` (async)
- **Caching**: `Redis`

**API Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/feed` | GET | Get personalized recommendations |
| `/api/article/{id}` | GET | Get article details |
| `/api/interaction` | POST | Log user interaction |
| `/api/profile` | GET | Get user preferences |
| `/api/similar/{id}` | GET | Find similar articles |
| `/api/topics` | GET | Get topic distribution |
| **`/api/chat`** | **POST** | **Send message to RAG assistant** |
| **`/api/chat/history`** | **GET** | **Get conversation history** |
| **`/api/chat/{conversation_id}`** | **DELETE** | **Clear conversation** |
| `/health` | GET | System health check |

**Frontend Features**:
- Feed view (list of articles)
- Article reader (click to read full text)
- **Chat interface (conversational assistant)**
- **Quick action buttons ("Summarize", "Explain", "Find related")**
- Like/skip buttons
- Topic filters
- Profile settings

**Interactions**:
- ← Receives requests from React frontend
- ↔ Queries PostgreSQL for articles/users
- ↔ Queries Redis for cached recommendations and conversations
- ↔ Queries FAISS for similar articles and RAG retrieval
- **↔ Calls LLM API for RAG generation**
- → Sends responses (JSON)
- → Triggers background tasks (Celery)

**Key Considerations**:
- Rate limiting (prevent abuse)
- Response time (<100ms for cached, <500ms uncached)
- Error handling and logging
- API versioning
- Auto-generated documentation (Swagger UI)

---

### 7. Infrastructure & Monitoring

**Purpose**: Deploy, scale, and monitor the entire system

**AWS Infrastructure**:

| Component | Service | Purpose |
|-----------|---------|---------|
| **Compute** | EC2 (t3.medium) | Run FastAPI + Celery workers |
| **Database** | RDS (PostgreSQL) | Persistent storage |
| **Cache** | ElastiCache (Redis) | Fast data access |
| **Storage** | S3 | Model artifacts, logs, backups |
| **Load Balancer** | ALB | Distribute traffic |
| **Monitoring** | CloudWatch | Metrics and alerts |
| **LLM** | OpenAI API / Ollama on GPU instance | RAG generation |

**Deployment Stack**:
- **Containerization**: Docker + docker-compose
- **Orchestration**: Docker Swarm or Kubernetes (optional)
- **CI/CD**: GitHub Actions
- **Infrastructure as Code**: Terraform (optional)

**Monitoring & Logging**:

| Tool | Purpose |
|------|---------|
| **CloudWatch** | System metrics (CPU, memory, disk) |
| **Custom Dashboards** | Business metrics (CTR, user engagement) |
| **Prometheus + Grafana** | Advanced metrics and alerting |
| **ELK Stack** | Log aggregation and search |
| **Sentry** | Error tracking |
| **WandB** | ML experiment tracking |

**Metrics to Track**:
- **API Performance**: Latency (p50, p95, p99), error rate
- **Recommendation Quality**: Click-through rate (CTR), diversity, coverage
- **Model Performance**: Accuracy, precision, recall, drift detection
- **System Health**: CPU/memory usage, queue length, cache hit rate
- **User Engagement**: DAU, session duration, articles per user
- **RAG Quality**: Response time, faithfulness score, citation accuracy, user satisfaction
- **RAG Costs**: LLM API tokens used, cost per conversation, cache hit rate

**CI/CD Pipeline**:
1. Developer pushes to GitHub
2. GitHub Actions runs tests
3. Build Docker image
4. Deploy to staging environment
5. Run integration tests
6. Manual approval
7. Deploy to production (blue-green deployment)

**Key Considerations**:
- Auto-scaling based on traffic
- Database backups (daily snapshots)
- Disaster recovery plan
- Cost optimization (use spot instances, right-size resources)
- Security (SSL, firewall rules, secret management)

---

## Data Flow

### Article Ingestion Flow
```
External Sources
    ↓
Celery Beat (Schedule)
    ↓
Scraper Task
    ↓
PostgreSQL (Raw Article)
    ↓
Processing Task (Celery)
    ↓
[Topic Classifier] + [Embedding Generator]
    ↓
PostgreSQL (Updated) + FAISS Index
    ↓
Ready for Recommendations
```

### User Interaction Flow
```
User clicks article
    ↓
React Frontend
    ↓
FastAPI POST /api/interaction
    ↓
PostgreSQL (Log interaction)
    ↓
Celery Task (Update profile)
    ↓
Redis (Cache updated profile)
    ↓
Influences future recommendations
```

### Recommendation Generation Flow
```
User requests feed
    ↓
FastAPI GET /api/feed
    ↓
Check Redis cache (hit = return cached)
    ↓
Cache miss → Recommendation Engine
    ↓
[Content-Based] + [Collaborative] + [Topic-Based] + [Popularity]
    ↓
Ensemble Combiner
    ↓
Post-processing (filter, diversify)
    ↓
Cache in Redis (30 min)
    ↓
Return to user
```

### RAG Conversation Flow
```
User asks question
    ↓
React Chat Component
    ↓
FastAPI POST /api/chat
    ↓
Load conversation history (Redis)
    ↓
Embed query (sentence-transformers)
    ↓
FAISS similarity search
    ↓
Retrieve top-K articles (PostgreSQL)
    ↓
Build context (query + articles + history)
    ↓
LLM API call (OpenAI / Ollama)
    ↓
Generate answer with citations
    ↓
Post-process & format
    ↓
Save to conversation memory (Redis + PostgreSQL)
    ↓
Return answer to user
    ↓
Frontend displays with source links
```

### Model Training Flow
```
Scheduled job (monthly) or manual trigger
    ↓
Export training data from PostgreSQL
    ↓
Google Colab or AWS SageMaker
    ↓
Fine-tune model
    ↓
Evaluate on test set
    ↓
Register in MLflow
    ↓
Deploy new model version
    ↓
A/B test vs. old model
    ↓
Rollout or rollback
```

---

## Tech Stack Summary

### Data Layer
```yaml
Primary Database: PostgreSQL 14+
  - Users, articles, interactions, profiles
  
Cache: Redis 7+
  - Recommendations, user profiles, sessions
  
Vector DB: FAISS (local) or Pinecone (cloud)
  - Article embeddings, similarity search
```

### Backend
```yaml
API Framework: FastAPI 0.100+
  - Async Python, auto docs, type hints
  
Task Queue: Celery 5+ + Redis
  - Background jobs, scheduled tasks
  
ORM: SQLAlchemy 2+
  - Database abstraction
  
Authentication: JWT (python-jose)
  - Secure user sessions
```

### Machine Learning
```yaml
Deep Learning: PyTorch 2+ + HuggingFace Transformers
  - Fine-tuning topic classifier
  
Embeddings: sentence-transformers
  - Pre-trained semantic embeddings
  
Collaborative Filtering: implicit
  - Matrix factorization (ALS)
  
Traditional ML: scikit-learn
  - Additional models, metrics

LLM Integration: OpenAI API / Ollama / Together.ai
  - RAG generation, conversational AI
  
RAG Framework: LangChain (optional) or custom
  - Orchestrate retrieval + generation
  
Experiment Tracking: WandB + MLflow
  - Track experiments, register models

RAG Evaluation: RAGAS
  - Measure RAG quality (faithfulness, relevance)
```

### Frontend
```yaml
Framework: React 18+ (or Next.js)
  - Modern UI components
  
Styling: Tailwind CSS
  - Utility-first CSS
  
HTTP Client: Axios
  - API communication
```

### Infrastructure
```yaml
Cloud: AWS
  - EC2, RDS, ElastiCache, S3, ALB
  
Containers: Docker + docker-compose
  - Consistent environments
  
CI/CD: GitHub Actions
  - Automated testing and deployment
  
Monitoring: CloudWatch, Prometheus, Grafana
  - System and business metrics
  
Logging: Python logging + ELK Stack
  - Centralized log management
```

### Development Tools
```yaml
Version Control: Git + GitHub
Package Management: pip + requirements.txt (or Poetry)
Code Quality: black, flake8, mypy
Testing: pytest + pytest-asyncio
Documentation: Sphinx or MkDocs
API Docs: Auto-generated with FastAPI (Swagger/ReDoc)
LLM Tools: OpenAI Python SDK, Ollama Python client, LangChain (optional)
```

---

## Deployment Architecture

### Development Environment
```
Local Machine
├── Docker Compose
│   ├── PostgreSQL container
│   ├── Redis container
│   ├── FastAPI container
│   └── Celery worker container
├── FAISS (local files)
└── Frontend (npm run dev)
```

### Staging Environment
```
AWS (Single Region)
├── EC2 Instance (t3.medium)
│   ├── FastAPI (gunicorn + 4 workers)
│   ├── Celery workers (4 workers)
│   └── Celery beat (scheduler)
├── RDS PostgreSQL (db.t3.micro)
├── ElastiCache Redis (cache.t3.micro)
└── S3 Bucket (models, logs)
```

### Production Environment
```
AWS (Multi-AZ)
├── Application Load Balancer
│   └── Health checks, SSL termination
├── Auto Scaling Group (2-4 instances)
│   ├── EC2 Instances (t3.medium)
│   │   ├── FastAPI (gunicorn + 8 workers)
│   │   └── Celery workers (8 workers)
├── RDS PostgreSQL (Multi-AZ, db.t3.small)
│   ├── Primary (read/write)
│   └── Standby (failover)
├── ElastiCache Redis Cluster
│   ├── Primary node
│   └── Read replicas
├── S3
│   ├── Model artifacts
│   ├── Application logs
│   └── Database backups
└── CloudWatch
    ├── Alarms
    └── Dashboards
```

### Scaling Strategy

**Horizontal Scaling**:
- Add more EC2 instances during peak hours
- Load balancer distributes traffic
- Stateless API (can scale freely)

**Vertical Scaling**:
- Increase database instance size as data grows
- Upgrade Redis for more memory

**Database Scaling**:
- Read replicas for heavy read workload
- Partition interactions table by date
- Archive old data to S3

**Cost Optimization**:
- Free Tier: PostgreSQL (RDS), Redis, EC2 (750 hrs/month)
- Reserved Instances: Save 30-50% for long-term
- Spot Instances: For Celery workers (70% cheaper)
- Auto-shutdown: Dev/staging environments at night

---

## Key Interactions & Data Flow Summary

### Critical Pathways

1. **Article → Recommendations**: 
   - Scraper → DB → Processor → FAISS → Recommender → API

2. **User Action → Better Recs**:
   - Click → API → DB → Profile Builder → Cache → Future Recs

3. **Model Training → Production**:
   - Dataset → Training → MLflow → Deploy → API

4. **Request → Response**:
   - User → Frontend → API → Cache/DB → Response

5. **RAG Conversation**:
   - Question → Embed → FAISS → Retrieve → LLM → Answer → User

### Asynchronous Processes

**Background Jobs (Celery)**:
- Scrape articles (every hour)
- Process articles (every 10 min)
- Rebuild user profiles (every hour)
- Retrain CF model (every day)
- Generate trending articles (every hour)

**Real-time Processes**:
- API requests (<500ms)
- Interaction logging (<50ms)
- Cache reads/writes (<10ms)

---

## Development Phases

### Phase 1: MVP (Weeks 1-3)
- Set up data collection (RSS scraping)
- Simple content-based recommendations (cosine similarity)
- Basic web interface
- Local deployment

### Phase 2: ML Enhancement (Weeks 4-6)
- Fine-tune topic classifier
- Add collaborative filtering
- Implement user tracking
- Improve UI

### Phase 3: RAG System (Weeks 7-9)
- **Implement basic RAG pipeline**
- **Integrate LLM (OpenAI or Ollama)**
- **Add chat interface**
- **Implement conversation memory**
- **Add citation system**

### Phase 4: Production (Weeks 10-12)
- Deploy to AWS
- Add caching and optimization
- Implement monitoring
- Load testing
- Response streaming

### Phase 5: Advanced Features (Weeks 13-15)
- A/B testing framework
- RAG evaluation metrics (RAGAS)
- Hybrid retrieval
- Advanced prompting
- Documentation

---

## Success Metrics

### Technical Metrics
- API latency: p95 < 200ms (recommendations), <3s (RAG)
- Cache hit rate: > 80%
- System uptime: > 99.5%
- Processing throughput: > 1000 articles/hour

### ML Metrics
- Topic classification accuracy: > 90%
- Recommendation diversity: > 0.7 (Gini coefficient)
- Coverage: Recommend > 80% of articles
- Model inference time: < 100ms
- **RAG faithfulness: > 0.9 (RAGAS)**
- **RAG citation accuracy: > 95%**

### Business Metrics
- Click-through rate (CTR): > 10%
- Session duration: > 5 minutes
- Articles per session: > 3
- User retention (7-day): > 30%
- **RAG feature adoption: > 30%**
- **Chat satisfaction: > 80% positive feedback**

---

## Interview Talking Points

When discussing this project in interviews, emphasize:

1. **End-to-end ML system**: From data collection to production deployment
2. **Multiple ML models**: Fine-tuning, collaborative filtering, embeddings
3. **Scalability**: Async processing, caching, background jobs
4. **Real-world challenges**: Cold start, data drift, A/B testing
5. **Full stack**: Backend, ML, infrastructure, monitoring
6. **Production-ready**: Error handling, logging, testing, CI/CD
7. **Business impact**: Can explain metrics and tradeoffs

**Key phrase**: "I built a production-grade recommendation system that combines fine-tuned transformers for topic classification with collaborative filtering, deployed on AWS with comprehensive monitoring."

---

## RAG System Deep Dive

### Why RAG for News?

**Traditional Approach Problems**:
- Users must browse through many articles to find information
- Hard to synthesize information across multiple sources
- Search only returns article lists, not answers
- No way to ask follow-up questions

**RAG Solution**:
- Direct answers to user questions
- Synthesizes information from multiple articles
- Provides citations for verification
- Conversational interface for exploration
- Always grounded in your article database (no hallucination)

### RAG Architecture Decisions

#### 1. LLM Choice

**Option A: OpenAI API (Recommended for MVP)**
```yaml
Model: gpt-4-turbo or gpt-3.5-turbo
Pros:
  - Best quality out of the box
  - No infrastructure needed
  - Fast inference
  - Good at following citation instructions
Cons:
  - Costs money ($0.01 per 1K tokens for GPT-3.5)
  - Data leaves your servers
  - Rate limits
Cost estimate: $50-200/month for moderate usage
```

**Option B: Local Models with Ollama**
```yaml
Models: Llama 3 8B, Mistral 7B, Mixtral 8x7B
Pros:
  - Free after setup
  - Data stays local
  - No rate limits
  - Full control
Cons:
  - Needs GPU (or slow on CPU)
  - Setup complexity
  - Slightly lower quality than GPT-4
  - Requires model management
Hardware: GPU with 8GB+ VRAM (or CPU with patience)
```

**Option C: Hosted Inference (Good Middle Ground)**
```yaml
Providers: Together.ai, Anyscale, Replicate
Pros:
  - Cheaper than OpenAI
  - More model choices
  - Better for scaling
  - No infrastructure
Cons:
  - Still costs money
  - Less mature than OpenAI
Cost estimate: $20-100/month
```

**Recommendation**: Start with OpenAI GPT-3.5-turbo (cheap, good quality), then optionally migrate to local or hosted for cost optimization.

#### 2. Retrieval Strategy

**Basic Semantic Search (MVP)**:
```python
# Simple but effective
query_embedding = embed_model.encode(user_query)
similar_articles = faiss_index.search(query_embedding, k=5)
```

**Enhanced Hybrid Search (Better)**:
```python
# Combine multiple signals
semantic_scores = faiss_search(query_embedding, k=20)
keyword_scores = bm25_search(query_tokens, k=20)
recency_scores = score_by_date(articles)

# Weighted combination
final_scores = (
    0.6 * semantic_scores +
    0.3 * keyword_scores +
    0.1 * recency_scores
)
top_articles = get_top_k(final_scores, k=5)
```

**Advanced Re-ranking (Production)**:
```python
# Use a cross-encoder to re-rank top-20
candidates = faiss_search(query_embedding, k=20)
reranker = CrossEncoder('ms-marco-MiniLM-L-12-v2')
scores = reranker.predict([(query, article.text) for article in candidates])
top_articles = get_top_k(scores, k=5)
```

#### 3. Prompt Engineering

**System Prompt Template**:
```
You are a helpful news assistant that answers questions based on provided 
articles. Follow these rules:

1. ONLY use information from the provided articles
2. ALWAYS cite your sources using [1], [2] format
3. If the articles don't contain relevant information, say so
4. Be concise but comprehensive
5. Maintain a neutral, journalistic tone
6. If asked about recent events, prioritize newer articles

Articles:
{retrieved_articles}

Conversation History:
{conversation_history}

User Question: {user_query}

Answer:
```

**Few-Shot Examples** (Optional but improves quality):
```
Example 1:
Question: What caused the stock market crash?
Answer: According to the articles, the crash was triggered by three factors: 
rising interest rates [1], tech sector selloff [2], and inflation concerns [1][3].

Example 2:
Question: What is quantum computing?
Answer: I don't have articles about quantum computing in the provided sources. 
Would you like me to search for articles on that topic?
```

#### 4. Conversation Memory

**Short-term Memory (Redis)**:
```python
# Store last N turns per user
conversation_key = f"chat:{user_id}:{conversation_id}"
redis.lpush(conversation_key, json.dumps({
    "role": "user",
    "content": user_query,
    "timestamp": now()
}))
redis.lpush(conversation_key, json.dumps({
    "role": "assistant", 
    "content": generated_answer,
    "sources": [article_ids],
    "timestamp": now()
}))
redis.ltrim(conversation_key, 0, 9)  # Keep last 10 messages
redis.expire(conversation_key, 3600)  # 1 hour TTL
```

**Long-term Storage (PostgreSQL)**:
```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    conversation_id UUID,
    role VARCHAR(20),
    content TEXT,
    sources JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Analytics queries
SELECT COUNT(*) FROM conversations WHERE created_at > NOW() - INTERVAL '7 days';
SELECT AVG(LENGTH(content)) FROM conversations WHERE role = 'assistant';
```

#### 5. Citation System

**During Generation**:
```python
# Build prompt with numbered articles
context = ""
for i, article in enumerate(retrieved_articles, 1):
    context += f"[{i}] Title: {article.title}\n"
    context += f"    Source: {article.source}\n"
    context += f"    Date: {article.published}\n"
    context += f"    Content: {article.text[:1000]}...\n\n"

prompt = system_prompt + context + user_query
```

**After Generation**:
```python
# Parse citations from response
import re

response = llm.generate(prompt)
citations = re.findall(r'\[(\d+)\]', response)

# Track which articles were used
used_articles = [retrieved_articles[int(c)-1] for c in citations]

# Log for analytics
log_sources(conversation_id, used_articles)
```

**Frontend Display**:
```javascript
// Show citations as clickable links
function formatResponse(text, articles) {
    return text.replace(/\[(\d+)\]/g, (match, num) => {
        const article = articles[num - 1];
        return `<a href="/article/${article.id}" class="citation">
                  [${num}]
                </a>`;
    });
}
```

### RAG Evaluation

**Automated Metrics (RAGAS)**:
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # Does answer match sources?
    answer_relevancy,      # Does answer address question?
    context_precision,     # Are retrieved docs relevant?
    context_recall         # Are all relevant docs retrieved?
)

# Evaluate on test set
results = evaluate(
    dataset=test_conversations,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

# Target scores (0-1 scale)
# faithfulness: > 0.9 (very important - no hallucination)
# answer_relevancy: > 0.8
# context_precision: > 0.7
# context_recall: > 0.7
```

**Human Evaluation**:
```python
# Collect feedback
@app.post("/api/chat/feedback")
def submit_feedback(conversation_id, message_id, rating, comment):
    # rating: 1-5 stars or thumbs up/down
    save_to_db({
        "conversation_id": conversation_id,
        "message_id": message_id,
        "rating": rating,
        "comment": comment,
        "timestamp": now()
    })
```

**A/B Testing Different Approaches**:
```python
# Route users to different RAG configurations
def get_rag_config(user_id):
    if user_id % 2 == 0:
        return RAGConfig(
            model="gpt-3.5-turbo",
            k=5,
            temperature=0.3,
            prompt_version="v1"
        )
    else:
        return RAGConfig(
            model="gpt-4-turbo",
            k=10,
            temperature=0.5,
            prompt_version="v2"
        )

# Track which performs better
track_metrics(user_id, config_version, response_time, user_rating)
```

### RAG Optimization

**1. Caching Common Queries**:
```python
# Cache popular questions
query_hash = hashlib.md5(query.lower().encode()).hexdigest()
cached = redis.get(f"rag_response:{query_hash}")
if cached:
    return json.loads(cached)

# Generate and cache
response = rag_pipeline(query)
redis.setex(f"rag_response:{query_hash}", 3600, json.dumps(response))
```

**2. Async Processing**:
```python
# Don't block on LLM call
@app.post("/api/chat")
async def chat(query: str):
    # Return immediately with conversation_id
    conversation_id = generate_id()
    
    # Process in background
    asyncio.create_task(
        process_rag_query(conversation_id, query)
    )
    
    return {"conversation_id": conversation_id, "status": "processing"}

# Client polls for response or use WebSockets for real-time
```

**3. Response Streaming**:
```python
# Stream tokens as they're generated
@app.post("/api/chat/stream")
async def chat_stream(query: str):
    async def generate():
        async for chunk in llm.stream(prompt):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**4. Smart Context Management**:
```python
# Don't exceed token limits
def build_context(query, articles, history, max_tokens=6000):
    # Reserve tokens
    query_tokens = count_tokens(query)
    history_tokens = count_tokens(history)
    reserved = query_tokens + history_tokens + 500  # 500 for response
    
    available = max_tokens - reserved
    
    # Fit articles in remaining space
    context = ""
    used_tokens = 0
    for article in articles:
        article_text = f"{article.title}\n{article.text[:1000]}"
        tokens = count_tokens(article_text)
        
        if used_tokens + tokens < available:
            context += article_text + "\n\n"
            used_tokens += tokens
        else:
            break
    
    return context
```

### Common RAG Challenges & Solutions

**Challenge 1: Hallucination**
- **Problem**: LLM makes up information not in sources
- **Solution**: Strong system prompt, lower temperature (0.1-0.3), citation requirement, evaluation metrics

**Challenge 2: Poor Retrieval**
- **Problem**: Relevant articles not retrieved
- **Solution**: Hybrid search (semantic + keyword), query expansion, re-ranking, increase K

**Challenge 3: Context Length**
- **Problem**: Too many articles exceed token limit
- **Solution**: Smart truncation, summarize articles first, use models with larger context (32K+)

**Challenge 4: Slow Response**
- **Problem**: Takes 5-10 seconds to respond
- **Solution**: Caching, streaming, async processing, faster models (GPT-3.5 vs GPT-4)

**Challenge 5: Incorrect Citations**
- **Problem**: LLM cites wrong sources or makes up citations
- **Solution**: Post-processing to verify citations, simpler citation format, examples in prompt

**Challenge 6: Off-Topic Responses**
- **Problem**: Answers questions outside of news domain
- **Solution**: System prompt constraints, query filtering, "I only answer questions about news"

### RAG Development Roadmap

**Week 1: Basic RAG**
- [ ] Set up OpenAI API or Ollama
- [ ] Implement basic retrieval (FAISS already done!)
- [ ] Write system prompt
- [ ] Generate first answers
- [ ] Simple chat UI

**Week 2: Quality Improvements**
- [ ] Add conversation memory (Redis)
- [ ] Implement citation system
- [ ] Add re-ranking
- [ ] Test with different prompts
- [ ] Collect feedback

**Week 3: Production Features**
- [ ] Response streaming
- [ ] Caching
- [ ] Error handling
- [ ] Logging and monitoring
- [ ] A/B testing framework

**Week 4: Advanced (Optional)**
- [ ] Hybrid retrieval
- [ ] Multi-turn query rewriting
- [ ] RAG evaluation metrics
- [ ] Cost optimization
- [ ] Documentation

### RAG Interview Talking Points

**Technical Depth**:
- "I implemented RAG using semantic search over 100K+ articles with FAISS"
- "I engineered prompts to ensure factual responses with proper citations"
- "I used temperature 0.3 and faithfulness metrics to prevent hallucination"
- "I optimized retrieval with hybrid search combining semantic and keyword matching"

**System Design**:
- "I designed the RAG pipeline to handle 100+ concurrent users with Redis caching"
- "I implemented response streaming for better UX while managing context windows"
- "I built conversation memory with Redis for short-term and PostgreSQL for analytics"

**Production Considerations**:
- "I tracked RAG quality with RAGAS metrics and human feedback"
- "I A/B tested different models and found GPT-3.5 offered the best cost/quality tradeoff"
- "I implemented caching which reduced LLM API costs by 70%"
- "I added citation verification to ensure responses are grounded in sources"

**ML Understanding**:
- "RAG is superior to fine-tuning for this use case because our news data updates daily"
- "I chose sentence-transformers for embeddings because they're specifically trained for semantic search"
- "I evaluated retrieval quality with precision@K and found K=5 optimal for response quality"

### Success Metrics for RAG

**Technical Metrics**:
- Response latency: <3 seconds (p95)
- Faithfulness score: >0.9 (RAGAS)
- Citation accuracy: >95%
- Cache hit rate: >60%

**User Metrics**:
- User satisfaction: >80% positive feedback
- Conversation length: >3 turns (engagement)
- Feature adoption: >30% of users try chat
- Return rate: >50% use chat again within 7 days

**Business Metrics**:
- Reduced bounce rate (users find answers faster)
- Increased session time (conversations keep users engaged)
- Lower support load (self-service Q&A)
- API cost per user: <$0.10/month

---

## Key Interview Talking Points