# Fine-tuning Topic Classification Model - Complete Guide

## Your Hardware Budget Analysis

### GPU Comparison
```yaml
A100 (20 hours available):
  - Memory: 40GB
  - Speed: 3-4x faster than T4
  - Best for: Large models (BERT-large, RoBERTa-large)
  - Cost equivalent: ~$2/hour = $40 value

T4 (100 hours available):
  - Memory: 16GB
  - Speed: Good for most models
  - Best for: Small-medium models (DistilBERT, BERT-base)
  - Cost equivalent: ~$0.50/hour = $50 value
```

### Recommendation: Use T4 for this project
**Why?**
- You have 100 hours (plenty for experimentation)
- Topic classification doesn't need huge models
- Save A100 hours for future projects (RAG, larger models)
- DistilBERT on T4 will train in ~30-45 minutes

---

## Model Selection

### Recommended Models (Best → Good)

#### 1. **DistilBERT-base-uncased** ⭐ BEST CHOICE
```yaml
Model: distilbert-base-uncased
Parameters: 66M
Training time: 30-45 min on T4
Inference: ~50ms per article
Accuracy: ~93-95% on AG News

Pros:
  - Fast training and inference
  - Small size (good for production)
  - Excellent accuracy for topic classification
  - 40% faster than BERT, 97% of performance
  
Cons:
  - Slightly less accurate than full BERT
  
HuggingFace: distilbert-base-uncased
```

#### 2. **BERT-base-uncased** (If you want max accuracy)
```yaml
Model: bert-base-uncased
Parameters: 110M
Training time: 60-90 min on T4
Inference: ~80ms per article
Accuracy: ~95-96% on AG News

Pros:
  - Higher accuracy
  - More robust
  - Standard baseline
  
Cons:
  - Slower than DistilBERT
  - Larger model size
  
HuggingFace: bert-base-uncased
```

#### 3. **RoBERTa-base** (Most accurate)
```yaml
Model: roberta-base
Parameters: 125M
Training time: 90-120 min on T4
Inference: ~90ms per article
Accuracy: ~96-97% on AG News

Pros:
  - Highest accuracy
  - Better pre-training
  - Robust to different text styles
  
Cons:
  - Slowest training/inference
  - Largest model
  
HuggingFace: roberta-base
```

#### 4. **DeBERTa-v3-small** (Good balance)
```yaml
Model: microsoft/deberta-v3-small
Parameters: 44M
Training time: 25-35 min on T4
Inference: ~40ms per article
Accuracy: ~94-95% on AG News

Pros:
  - Smallest model
  - Very fast
  - Good accuracy
  
Cons:
  - Less widely known
  - Slightly lower accuracy than DistilBERT
  
HuggingFace: microsoft/deberta-v3-small
```

### My Recommendation: **DistilBERT-base-uncased**
- Perfect balance of speed, accuracy, and size
- Industry standard for text classification
- Will finish training in 30-45 minutes
- Easy to deploy in production

---

## Dataset Selection

### Option 1: **AG News** ⭐ RECOMMENDED
```yaml
Dataset: ag_news
Size: 120,000 articles
Classes: 4 (World, Sports, Business, Tech/Sci)
Split: 120K train, 7.6K test
Source: HuggingFace datasets

Pros:
  - Perfect for news classification
  - Clean, well-balanced data
  - Fast to download and process
  - Matches your use case exactly
  
Cons:
  - Only 4 categories (might want more granularity)
  
Load with:
  from datasets import load_dataset
  dataset = load_dataset("ag_news")
```

### Option 2: **20 Newsgroups** (Alternative)
```yaml
Dataset: SetFit/20_newsgroups
Size: ~20,000 articles
Classes: 20 (more granular topics)
Split: Standard train/test

Pros:
  - More categories
  - Includes tech subcategories
  
Cons:
  - Smaller dataset
  - Some categories not news-related
  - Older data (1990s)
  
When to use: If you want finer-grained classification
```

### Option 3: **BBC News** (Small but clean)
```yaml
Dataset: SetFit/bbc-news
Size: 2,225 articles
Classes: 5 (Business, Entertainment, Politics, Sport, Tech)
Split: Need to create your own

Pros:
  - Very clean data
  - Modern news articles
  
Cons:
  - Very small (need to augment or combine)
  
When to use: As validation set or for British English
```

### Option 4: **Custom Combined Dataset** (Advanced)
```python
# Combine multiple datasets for more training data
from datasets import load_dataset, concatenate_datasets

ag_news = load_dataset("ag_news")
bbc = load_dataset("SetFit/bbc-news")

# Map BBC labels to AG News categories
# Then concatenate for more training data
```

### My Recommendation: Start with **AG News**
- 120K samples is perfect (not too small, not too large)
- Clean, balanced data
- Trains quickly
- Can always add more data later

---

## Fine-tuning Strategy

### Hyperparameters (Optimized for T4)

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Output
    output_dir="./results",
    
    # Training duration
    num_train_epochs=3,              # 3 epochs is standard
    
    # Batch size (adjust based on GPU memory)
    per_device_train_batch_size=32,  # T4: 32 works well
    per_device_eval_batch_size=64,   # Eval can be larger
    
    # Optimizer
    learning_rate=2e-5,              # Standard for BERT models
    weight_decay=0.01,               # L2 regularization
    
    # Learning rate schedule
    warmup_steps=500,                # Gradual warmup
    lr_scheduler_type="linear",      # Linear decay
    
    # Evaluation
    eval_strategy="steps",           # Evaluate during training
    eval_steps=500,                  # Every 500 steps
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    
    # Performance
    fp16=True,                       # Mixed precision (2x faster on T4)
    dataloader_num_workers=2,        # Parallel data loading
    
    # Logging
    logging_dir="./logs",
    logging_steps=100,
    report_to="wandb",               # Track experiments
    
    # Other
    save_total_limit=2,              # Keep only 2 checkpoints
    push_to_hub=False,               # Set True to upload to HF
)
```

### Training Time Estimates

| Model | GPU | Batch Size | Epochs | Time |
|-------|-----|------------|--------|------|
| DistilBERT | T4 | 32 | 3 | ~30-45 min |
| DistilBERT | A100 | 64 | 3 | ~10-15 min |
| BERT-base | T4 | 16 | 3 | ~60-90 min |
| BERT-base | A100 | 32 | 3 | ~20-30 min |
| RoBERTa | T4 | 16 | 3 | ~90-120 min |

### Expected Results

| Model | AG News Accuracy | F1 Score |
|-------|------------------|----------|
| DistilBERT | 93-95% | 0.93-0.95 |
| BERT-base | 95-96% | 0.95-0.96 |
| RoBERTa | 96-97% | 0.96-0.97 |
| DeBERTa-small | 94-95% | 0.94-0.95 |

---

## Complete Training Pipeline

### Setup (5 minutes)
```python
# Install packages
!pip install transformers datasets accelerate wandb evaluate scikit-learn

# Login to track experiments
import wandb
wandb.login()  # Get key from wandb.ai

# GPU check
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Load Data (2 minutes)
```python
from datasets import load_dataset

# Load AG News
dataset = load_dataset("ag_news")

# Check data
print(dataset)
print(dataset['train'][0])
# Output: {'text': 'Wall St. Bears Claw Back...', 'label': 2}

# Labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
```

### Tokenize (3 minutes)
```python
from transformers import AutoTokenizer

# Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256  # News titles/snippets are short
    )

# Apply to entire dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=2,  # Parallel processing
    remove_columns=["text"]  # Remove original text
)

# Rename label column
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
```

### Create Model (1 minute)
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,  # 4 categories
    id2label={0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'},
    label2id={'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}
)
```

### Define Metrics (1 minute)
```python
import evaluate
import numpy as np

# Load metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )
    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average='weighted'
    )
    
    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1']
    }
```

### Train! (30-45 minutes)
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./ag_news_distilbert",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    report_to="wandb",
    run_name="ag_news_distilbert_v1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Start training!
trainer.train()
```

### Evaluate (2 minutes)
```python
# Final evaluation
results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"F1 Score: {results['eval_f1']:.4f}")

# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

predictions = trainer.predict(tokenized_datasets["test"])
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = tokenized_datasets["test"]["labels"]

print("\nClassification Report:")
print(classification_report(
    y_true, 
    y_pred, 
    target_names=label_names
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
```

### Save Model (2 minutes)
```python
# Save to local
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

# Download to your computer
from google.colab import files

# Create archive
!zip -r final_model.zip ./final_model

# Download (optional - can also save to Google Drive)
files.download('final_model.zip')

# Or save to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r ./final_model /content/drive/MyDrive/news_recommender/
```

---

## Optimization Tips

### 1. Speed Up Training
```python
# Use gradient accumulation if memory is tight
training_args = TrainingArguments(
    per_device_train_batch_size=16,  # Smaller batch
    gradient_accumulation_steps=2,   # Accumulate 2 steps = effective batch 32
    ...
)

# Use mixed precision
training_args = TrainingArguments(
    fp16=True,  # Always use this on T4/A100
    ...
)
```

### 2. Prevent Overfitting
```python
training_args = TrainingArguments(
    weight_decay=0.01,              # L2 regularization
    warmup_steps=500,               # Gradual learning rate increase
    eval_strategy="steps",          # Monitor validation
    early_stopping_patience=3,      # Stop if no improvement
    ...
)
```

### 3. Hyperparameter Tuning (Advanced)
```python
# Try different learning rates
learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]

for lr in learning_rates:
    training_args = TrainingArguments(
        learning_rate=lr,
        run_name=f"ag_news_lr_{lr}",
        ...
    )
    trainer = Trainer(...)
    trainer.train()
```

---

## Complete Colab Notebook Structure

### Time Budget Breakdown
```
Setup & Installation:     5 min
Data Loading:            2 min
Tokenization:            3 min
Model Creation:          1 min
Training (3 epochs):    30-45 min
Evaluation:              2 min
Saving:                  2 min
------------------------------
TOTAL:                  45-60 min
```

### Notebook Sections
```markdown
# 1. Setup (Run Once)
- Install packages
- Import libraries
- Check GPU
- Login to WandB

# 2. Data Preparation
- Load AG News dataset
- Explore data
- Create train/val/test splits

# 3. Tokenization
- Load tokenizer
- Tokenize all data
- Create dataloaders

# 4. Model Definition
- Load pre-trained model
- Define training args
- Create trainer

# 5. Training
- Train model
- Monitor metrics
- Save checkpoints

# 6. Evaluation
- Test set performance
- Confusion matrix
- Error analysis

# 7. Save & Export
- Save model
- Download or save to Drive
- Export for production
```

---

## What You Should Try (Experimentation Plan)

### Experiment 1: Baseline (1 hour)
```yaml
Model: distilbert-base-uncased
Dataset: AG News (full)
Epochs: 3
Learning rate: 2e-5
Batch size: 32

Goal: Get baseline accuracy (~94%)
GPU: T4
```

### Experiment 2: Bigger Model (1.5 hours)
```yaml
Model: bert-base-uncased
Same settings as above

Goal: Compare accuracy vs DistilBERT
GPU: T4
```

### Experiment 3: Learning Rate Tuning (3 hours)
```yaml
Model: distilbert-base-uncased
Learning rates: [1e-5, 2e-5, 3e-5, 5e-5]

Goal: Find optimal learning rate
GPU: T4
```

### Experiment 4: More Epochs (if needed)
```yaml
If accuracy < 93%:
  Try 5 epochs instead of 3
  Reduce learning rate to 1e-5
  Add more warmup steps
```

### Total GPU Time: ~5-7 hours (out of your 100 T4 hours)

---

## Success Criteria

### Minimum (Good Enough)
- ✅ Accuracy: > 92%
- ✅ Training time: < 1 hour
- ✅ Model size: < 500MB
- ✅ Inference: < 100ms per article

### Target (Great)
- ✅ Accuracy: > 94%
- ✅ F1 score: > 0.94
- ✅ All classes: > 90% accuracy
- ✅ No severe class imbalance

### Excellent (Impressive)
- ✅ Accuracy: > 95%
- ✅ Balanced across all topics
- ✅ Documented experiments in WandB
- ✅ Model deployed and working

---

## Common Issues & Solutions

### Issue 1: Out of Memory
```python
# Solution: Reduce batch size
per_device_train_batch_size=16  # Instead of 32

# Or use gradient accumulation
gradient_accumulation_steps=2
```

### Issue 2: Low Accuracy (<90%)
```python
# Solutions:
1. Train for more epochs (5 instead of 3)
2. Lower learning rate (1e-5)
3. Check data preprocessing
4. Try different model (BERT instead of DistilBERT)
```

### Issue 3: Training Too Slow
```python
# Solutions:
1. Enable fp16=True (2x faster)
2. Increase batch size
3. Reduce max_length in tokenization
4. Use DistilBERT instead of BERT
```

### Issue 4: Overfitting
```python
# Symptoms: Train accuracy >> Test accuracy
# Solutions:
1. Add weight_decay=0.01
2. Reduce epochs
3. Add dropout (model.config.hidden_dropout_prob = 0.2)
4. Get more data
```

---

## Next Steps After Training

1. **Save to Google Drive**
   - Don't lose your trained model!
   - Backup to Drive immediately

2. **Export for Production**
   - Save in format compatible with FastAPI
   - Test inference speed locally

3. **Document Results**
   - WandB has all your metrics
   - Screenshot best results
   - Note hyperparameters used

4. **Integrate into Pipeline**
   - Load model in your article processor
   - Test on real news articles
   - Measure inference latency

---

## Ready to Start?

I can provide you with:
1. **Complete Colab notebook** (copy-paste ready)
2. **Starter code** (minimal version)
3. **Full pipeline** (with all experiments)

Which would you prefer?