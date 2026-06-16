from transformers import pipeline

labels = [
    "Technical issue",
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request"
]

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

ticket = input(
    "Enter Ticket:\n"
)

result = classifier(
    ticket,
    labels
)

print("\nTop 3 Tags:\n")

for label, score in zip(
    result["labels"][:3],
    result["scores"][:3]
):
    print(
        f"{label} : {score:.4f}"
    )