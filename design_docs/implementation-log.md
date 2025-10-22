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
- 

**Dev commands**
1. .\venv\Scripts\activate
2. uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
3. docker-compose up -d
4. docker-compose down