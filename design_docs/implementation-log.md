# Personal News Recommender Implementation Log

## Project Overview

TestGen-LLM is an implementation of the approach described in the paper "Automated Unit Test Improvement using Large Language Models at Meta." The system follows the Assured LLMSE approach, providing verifiable guarantees about the improvements made to test classes.

## Developer's Diary

### [Date: 2025-10-18]

- Starting with the finetuning
- **Why am i finetuning?**
    - for faster inference time using smaller model
    - higher accuracy
    - obv interview

- Model choice: **DistilBERT-base-uncased**
    - distilled from BERT-base (110M) => 67M
    - uncased, no capitalization, standard for classification

- Training in Google Colab Pro using T4 GPU

- 94.81% accuracy, about 40 minutes of training

### [Date: 2025-10-19]

- setup the project structure
- settings, basic API setup
    - localhost:8000 works

- docker-compose

### [Date: 2025-10-20]

- you gotta open docker desktop to run the dockers lol
- scripts/test_connection.py runs successfully
- finished setup.md aight cool good stuff

### [Date: 2025-10-21]

- all basic APIs working
- interactions won't work without any existing user

### [Date: 2025-10-22]

- **Implemented Topic Classifier Module** ✓
    - Created `backend/ml/classifier.py` with TopicClassifier class
    - Singleton pattern, thread-safe, production-ready
    - Supports batch processing (276 texts/second on CPU)
    - Inference time: ~3.6ms per article

- **Added Classification API Endpoints** ✓
    - POST `/api/classify/text` - Classify raw text
    - POST `/api/classify/article/{id}` - Classify article by ID
    - POST `/api/classify/batch` - Batch classification (up to 100 texts)
    - GET `/api/classify/model-info` - Model metadata
    - All endpoints tested and working

- **Deployment-Friendly Design**
    - Model path configurable via `.env` (TOPIC_CLASSIFIER_PATH)
    - Easy to switch between local/S3/cloud storage
    - Auto device detection (CPU/GPU)

- **Test Scripts Created**
    - `scripts/test_classifier.py` - Unit tests for classifier
    - `scripts/test_classify_api.py` - API endpoint tests
    - All tests passing

- `__init__.py` specificies that the folder is a package
    - `__all__` specifies what's public API
    - when importing, doesn't need to know exact file structures
    - i.e.  `from backend.ml.embedder import ...` to \
        `from backend.ml import ...`

- implemented FAISS with basic operations
    - singleton, consistent with classifier and embedder

**To do**
1. ~~Load & test your fine-tuned model (create backend/ml/classifier.py)~~
2. ~~Create embedding generator (create backend/ml/embedder.py)~~
3. ~~Set up FAISS (create backend/ml/vector_store.py)~~
4. Build processing script (create backend/processors/article_processor.py)
5. Add recommendation endpoint to main.py

**Future improvements**
- FAISS save timings (after batch? periodic?)

**Dev commands**
1. .\venv\Scripts\activate
2. uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
3. docker-compose up -d
4. docker-compose down