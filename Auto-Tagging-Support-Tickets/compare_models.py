import pandas as pd

comparison = pd.DataFrame({
    "Method": [
        "Zero Shot",
        "Few Shot"
    ],
    "Description": [
        "Classification without examples",
        "Classification using examples"
    ]
})

print(comparison)

comparison.to_csv(
    "outputs/comparison_results.csv",
    index=False
)

print("\nSaved: outputs/comparison_results.csv")