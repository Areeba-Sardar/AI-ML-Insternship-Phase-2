from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load saved model
model_path = "model/news_classifier"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Labels
labels = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Test text
text = "Tesla launches a new AI chip for autonomous driving"

# Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

# Prediction
with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=1).item()

print("\nNews:")
print(text)

print("\nPredicted Category:")
print(labels[prediction])