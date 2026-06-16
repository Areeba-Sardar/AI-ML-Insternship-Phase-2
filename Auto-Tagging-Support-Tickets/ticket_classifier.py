import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score

print("Loading Dataset...")

df = pd.read_csv("data/customer_support_tickets.csv")

df = df[
    [
        "Ticket Type",
        "Ticket Subject",
        "Ticket Description"
    ]
]

df = df.dropna()

df["ticket_text"] = (
    df["Ticket Subject"].astype(str)
    + " "
    + df["Ticket Description"].astype(str)
)

candidate_labels = [
    "Technical issue",
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request"
]

print("\nLoading Zero-Shot Model...")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Test on 20 rows
sample_df = df.head(20).copy()

predictions = []

print("\nPredicting...\n")

for i, text in enumerate(sample_df["ticket_text"], start=1):

    print(f"Processing {i}/{len(sample_df)}")

    result = classifier(
        text,
        candidate_labels
    )

    predictions.append(
        result["labels"][0]
    )

sample_df["Predicted_Tag"] = predictions

print("\nActual vs Predicted\n")

print(
    sample_df[
        [
            "Ticket Type",
            "Predicted_Tag"
        ]
    ]
)

accuracy = accuracy_score(
    sample_df["Ticket Type"],
    sample_df["Predicted_Tag"]
)

print(f"\nZero-Shot Accuracy: {accuracy:.4f}")

# Top 3 Tags Example
sample_result = classifier(
    sample_df["ticket_text"].iloc[0],
    candidate_labels
)

print("\nTop 3 Tags For First Ticket:\n")

for label, score in zip(
    sample_result["labels"][:3],
    sample_result["scores"][:3]
):
    print(
        f"{label} : {score:.4f}"
    )

sample_df.to_csv(
    "outputs/zero_shot_predictions.csv",
    index=False
)

print("\nSaved: outputs/zero_shot_predictions.csv")