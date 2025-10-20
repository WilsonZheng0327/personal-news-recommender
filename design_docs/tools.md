# News Recommender System - Complete Tools & Services Guide

## Overview: Free Local Development → Optional Paid Production

**Good news**: You can build and test **everything locally for FREE**. Paid services are only needed if you want to deploy to production or use premium LLMs.

---

## Development Environment (100% Free)

### Required Tools

#### 1. **Python 3.10+**
```bash
# Check if installed
python --version

# Install (if needed)
# macOS: brew install python
# Ubuntu: sudo apt install python3.10
# Windows: Download from python.org
```
**Cost**: Free
**Purpose**: Programming language for entire backend

#### 2. **Git & GitHub**
```bash
# Check if installed
git --version

# Create account at github.com (free)
```
**Cost**: Free
**Purpose**: Version control, code hosting

#### 3. **Docker Desktop**
```bash
# Download from docker.com
# Includes docker-compose
```
**Cost**: Free for personal use
**Purpose**: Run PostgreSQL, Redis locally without installation headaches

#### 4. **VS Code** (or any IDE)
```bash
# Download from code.visualstudio.com
```
**Cost**: Free
**Purpose**: Code editor with Python extensions

---

## Core Services: Local vs Cloud

### 1. Database (PostgreSQL)

#### Local Development (FREE)
```yaml
Option A: Docker (Recommended)
  Command: docker run -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:14
  Cost: Free
  Pros: Easy setup, isolated, delete anytime
  Cons: Data lost when container removed (use volumes for persistence)

Option B: Direct Install
  macOS: brew install postgresql
  Ubuntu: sudo apt install postgresql
  Windows: Download installer
  Cost: Free
  Pros: Persistent data
  Cons: Harder to clean up
```

#### Cloud Production (PAID - but not needed initially)
```yaml
AWS RDS:
  Cost: $15-50/month (db.t3.micro free tier: 750 hrs/month for 12 months)
  Setup: AWS account → RDS → Create PostgreSQL instance
  
Alternatives:
  - Supabase: Free tier (500MB database, 2GB bandwidth)
  - Railway: Free tier ($5 credit, then $5/month)
  - Neon: Free tier (1 database, 10GB storage)
  
Recommendation: Use local for development, Supabase/Railway for first deployment
```

### 2. Cache (Redis)

#### Local Development (FREE)
```yaml
Docker (Recommended):
  Command: docker run -p 6379:6379 redis:7
  Cost: Free
  
Direct Install:
  macOS: brew install redis
  Ubuntu: sudo apt install redis
  Windows: Use Docker
```

#### Cloud Production (PAID - optional)
```yaml
AWS ElastiCache:
  Cost: $15-30/month (cache.t3.micro)
  
Free Alternatives:
  - Upstash: Free tier (10,000 commands/day)
  - Redis Cloud: Free tier (30MB)
  
Recommendation: Upstash free tier is perfect for testing production
```

### 3. Vector Database (FAISS)

#### Local Development (FREE)
```bash
pip install faiss-cpu  # For CPU
# OR
pip install faiss-gpu  # If you have NVIDIA GPU

# Store index as file on disk
import faiss
index = faiss.IndexFlatL2(384)  # 384-dim vectors
# ... add vectors ...
faiss.write_index(index, "articles.index")
```
**Cost**: Free
**Storage**: Local filesystem
**Scalability**: Can handle 100K-1M vectors easily on laptop

#### Cloud Production (PAID - optional)
```yaml
Pinecone:
  Cost: Free tier (1 index, 100K vectors, 5GB storage)
  Paid: $70/month for more
  Pros: Managed, scalable, easy API
  Cons: Expensive for hobby projects
  
Weaviate (Self-hosted):
  Cost: Free (run on your own server)
  Needs: EC2 instance (~$10/month)
  
Recommendation: Stick with FAISS locally, upgrade to Pinecone only if needed
```

---

## ML & AI Services

### 4. LLM for RAG System

#### Option A: OpenAI API (PAID - Recommended for quality)
```yaml
Setup:
  1. Create account at platform.openai.com
  2. Add payment method
  3. Get API key
  
Models:
  - gpt-3.5-turbo: $0.50 per 1M input tokens, $1.50 per 1M output
  - gpt-4-turbo: $10 per 1M input tokens, $30 per 1M output
  - gpt-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output (BEST VALUE)
  
Cost Estimate:
  - Light testing: $5-10/month
  - Moderate use: $20-50/month
  - Heavy use: $100+/month
  
Tips to Save Money:
  - Use gpt-4o-mini (cheaper, still great)
  - Cache common queries
  - Set monthly spending limits
  - Use for production only, Ollama for testing
```

#### Option B: Ollama (FREE - Local Models)
```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Download models
ollama pull llama3.2:3b      # 2GB, fast
ollama pull llama3.1:8b      # 4.7GB, better quality
ollama pull mistral:7b       # 4.1GB, good balance

# Use in Python
pip install ollama

from ollama import Client
client = Client()
response = client.chat(model='llama3.1:8b', messages=[
    {'role': 'user', 'content': 'Hello!'}
])
```
**Cost**: Free
**Hardware**: Works on CPU (slow) or GPU (fast)
**Quality**: Good (80-90% of GPT-3.5), excellent for testing
**Pros**: No API costs, private, unlimited usage
**Cons**: Slower, needs disk space, slightly lower quality

#### Option C: Free/Cheap Hosted APIs
```yaml
Together.ai:
  Cost: $0.20 per 1M tokens (4x cheaper than OpenAI)
  Models: Llama, Mistral, many open-source options
  Free tier: $25 credit
  
Groq:
  Cost: Free tier (generous limits)
  Speed: VERY fast inference
  Models: Llama, Mixtral
  
Hugging Face Inference API:
  Cost: Free tier available
  Models: Any open-source model
  
Recommendation: Start with Ollama for development, OpenAI gpt-4o-mini for production
```

### 5. Model Training (Fine-tuning Topic Classifier)

#### Local Training (FREE)
```yaml
Your Laptop:
  Requirements: 8GB+ RAM
  Time: 2-3 hours on CPU, 30 min on GPU
  Cost: Free, just electricity
  Suitable for: AG News dataset (small)
```

#### Cloud Training (FREE with limits)
```yaml
Google Colab:
  Cost: FREE tier (T4 GPU for 12 hours/day)
  Paid: Colab Pro $10/month (better GPUs, longer sessions)
  Setup: colab.research.google.com
  Pros: Free GPU, no setup
  Cons: Session timeouts, limited hours
  
Kaggle Notebooks:
  Cost: FREE (30 hrs/week GPU)
  Similar to Colab
  
Recommendation: Use FREE Colab for initial training
```

#### Paid Training (optional, for advanced)
```yaml
AWS SageMaker:
  Cost: $0.05-1.00/hour depending on instance
  Free tier: 250 hours/month for 2 months (new accounts)
  
Lambda Labs:
  Cost: $0.50-1.50/hour for GPU instances
  Cheaper than AWS
  
Recommendation: Not needed unless doing very large models
```

### 6. Experiment Tracking

#### Weights & Biases (WandB)
```bash
pip install wandb
wandb login  # Free account at wandb.ai
```
**Cost**: Free tier (100GB storage, unlimited runs)
**Paid**: $50/month for teams (not needed)

#### MLflow (Self-hosted)
```bash
pip install mlflow
mlflow ui  # Runs locally
```
**Cost**: Free (stores data locally)

**Recommendation**: Start with free WandB, it's excellent

---

## Cloud Platforms (All optional for development)

### AWS (Most Popular)
```yaml
Free Tier (12 months for new accounts):
  - EC2: 750 hours/month (t2.micro)
  - RDS: 750 hours/month (db.t2.micro)
  - S3: 5GB storage
  - Lambda: 1M free requests/month
  
After Free Tier (~$50-100/month for full system):
  - EC2 t3.medium: ~$30/month
  - RDS db.t3.micro: ~$15/month
  - ElastiCache: ~$15/month
  - S3: ~$5/month
  - Load Balancer: ~$20/month
  
Recommendation: Use free tier for learning, pause when not testing
```

### Railway (Easiest)
```yaml
Cost: $5/month (500 hrs compute)
Pros: 
  - Very easy deployment
  - PostgreSQL, Redis included
  - Good for small projects
Cons:
  - More expensive than AWS at scale
  
Recommendation: BEST for first deployment
```

### Render
```yaml
Cost: 
  - Free tier (services sleep after inactivity)
  - $7/month for always-on service
Pros: Easy, good free tier
Cons: Free tier is slow

Recommendation: Good for demos
```

### Fly.io
```yaml
Cost: Free tier (3 shared VMs, 3GB storage)
Pros: Generous free tier, global deployment
Cons: Slight learning curve

Recommendation: Great free option
```

---

## Development Tools & Libraries (All FREE)

### Python Packages
```bash
# Install all at once
pip install \
  # Web Framework
  fastapi uvicorn \
  # Database
  sqlalchemy psycopg2-binary alembic \
  # Cache
  redis \
  # Task Queue
  celery \
  # ML Core
  torch transformers sentence-transformers \
  # Vector Search
  faiss-cpu \
  # Collaborative Filtering
  implicit \
  # Data Processing
  pandas numpy scipy \
  # Web Scraping
  feedparser newspaper3k beautifulsoup4 requests \
  # LLM
  openai ollama langchain \
  # Testing
  pytest pytest-asyncio \
  # Utilities
  python-dotenv pydantic
```

### Additional Tools
```yaml
Postman / Thunder Client:
  Cost: Free
  Purpose: Test API endpoints
  
DBeaver / pgAdmin:
  Cost: Free
  Purpose: Database GUI
  
Redis Insight:
  Cost: Free
  Purpose: Redis GUI
```

---

## Recommended Setup Path

### Phase 1: Pure Local (Week 1-8) - $0
```yaml
What you need:
  - Python, Docker, VS Code (all free)
  - Docker containers: PostgreSQL, Redis
  - FAISS (local file storage)
  - Ollama (local LLM)
  - All Python packages (free)
  - WandB free tier
  
Total cost: $0/month
Can build: Entire system, test everything
```

### Phase 2: Add LLM Quality (Week 9+) - ~$10/month
```yaml
Add:
  - OpenAI API (gpt-4o-mini)
  - Set spending limit to $10/month
  
Total cost: ~$10/month
Benefit: Much better RAG responses
```

### Phase 3: First Deployment (Week 12+) - ~$5-15/month
```yaml
Option A: Railway
  - Deploy backend: $5/month
  - PostgreSQL + Redis included
  - Total: $5/month
  
Option B: Fly.io
  - Use free tier: $0/month
  - Limitations: Sleeps when inactive
  
Option C: AWS Free Tier
  - All services free for 12 months
  - After: ~$50/month (can reduce costs)
  
Recommendation: Railway for easiest start
```

### Phase 4: Production Ready (When needed) - ~$50-150/month
```yaml
Full AWS Setup:
  - EC2 + RDS + ElastiCache: ~$60/month
  - OpenAI API: ~$30/month
  - Pinecone (optional): ~$70/month OR stay with FAISS
  - Domain + SSL: ~$12/year
  
Total: ~$90/month (without Pinecone)

Cost Optimizations:
  - Use spot instances (70% cheaper)
  - Turn off dev/staging at night
  - Cache aggressively (reduce LLM calls)
  - Use Ollama for some queries
```

---

## What I Recommend for You

### Immediate (Next 2 weeks): $0
```bash
# 1. Install local tools
brew install python git docker  # macOS
# or equivalent for your OS

# 2. Start Docker containers
docker-compose up -d  # I'll give you the config

# 3. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# 4. Install Python packages
pip install -r requirements.txt  # I'll create this

# 5. Create free accounts
- GitHub (code hosting)
- WandB (experiment tracking)
- Google Colab (GPU training)

Total cost: $0
Time: 1-2 hours setup
```

### After Building MVP (Week 9): ~$10/month
```bash
# Add OpenAI for better RAG quality
1. Create OpenAI account
2. Add payment method
3. Set $10 monthly limit
4. Use gpt-4o-mini model

This gives you production-quality responses while developing
```

### When Ready to Deploy (Week 12): ~$5/month
```bash
# Deploy to Railway
1. Create Railway account
2. Connect GitHub repo
3. Deploy with one click
4. Share with friends/recruiters

Your project is now live!
```

---

## Cost Comparison Table

| Stage | Services | Monthly Cost | What You Get |
|-------|----------|--------------|--------------|
| **Local Dev** | Docker, Ollama, Python | $0 | Full development environment |
| **+ Good LLM** | + OpenAI gpt-4o-mini | ~$10 | Better RAG responses |
| **First Deploy** | Railway | ~$5 | Live website, PostgreSQL, Redis |
| **Full Production** | AWS (EC2, RDS, etc) | ~$60 | Scalable, professional setup |
| **Enterprise** | + Pinecone, monitoring | ~$150+ | Production-grade system |

---

## What You DON'T Need (Common Misconceptions)

❌ **Paid cloud platform** - Local dev is totally fine
❌ **Pinecone** - FAISS works great locally
❌ **OpenAI API** - Ollama is free and good enough for testing
❌ **Expensive GPU** - Colab is free, CPU works for small models
❌ **Multiple services** - Docker handles everything locally
❌ **MLflow** - WandB free tier is better and easier

---

## Quick Start Shopping List

### Absolute Minimum ($0)
- [ ] Python 3.10+
- [ ] Docker Desktop
- [ ] VS Code
- [ ] GitHub account (free)
- [ ] Google Colab account (free)

### Recommended ($10/month)
- [ ] Everything above
- [ ] OpenAI API account (for RAG quality)
- [ ] Set $10 spending limit

### When Ready to Show Off ($5-15/month)
- [ ] Everything above  
- [ ] Railway account (easiest deployment)
- [ ] Custom domain (optional, ~$12/year)

---

## Next Steps

Would you like me to create:
1. **docker-compose.yml** - One command to start PostgreSQL + Redis
2. **requirements.txt** - All Python packages you need
3. **.env.example** - Template for API keys and configs
4. **Detailed setup guide** - Step-by-step from zero to running

Just let me know and I'll generate these files for you!