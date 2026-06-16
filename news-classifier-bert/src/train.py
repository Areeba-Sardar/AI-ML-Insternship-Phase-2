from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np

# =====================================
# 1. LOAD DATASET
# =====================================

print("Loading dataset...")

dataset = load_dataset("ag_news")

# Small subset for faster training on CPU
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(10000))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(2000))

print(dataset)

# =====================================
# 2. LOAD TOKENIZER
# =====================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# =====================================
# 3. TOKENIZATION
# =====================================

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

print("Tokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

# =====================================
# 4. FORMAT DATASET
# =====================================

tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# =====================================
# 5. LOAD MODEL
# =====================================

print("Loading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# =====================================
# 6. METRICS
# =====================================

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
        average="weighted"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

# =====================================
# 7. TRAINING ARGUMENTS
# =====================================

training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True
)

# =====================================
# 8. TRAINER
# =====================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)

# =====================================
# 9. TRAIN
# =====================================

print("Training started...")

trainer.train()

# =====================================
# 10. EVALUATE
# =====================================

print("Evaluating model...")

results = trainer.evaluate()

print("\nResults:")
print(results)

# =====================================

# 11. SAVE MODEL
# =====================================

save_path = "model/news_classifier"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\nModel saved at: {save_path}")
print("Task 1 Training Completed Successfully!")