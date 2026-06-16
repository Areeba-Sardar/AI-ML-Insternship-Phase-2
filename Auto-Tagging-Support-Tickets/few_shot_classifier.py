import pandas as pd

df = pd.read_csv(
    "data/customer_support_tickets.csv"
)

df = df.dropna()

ticket = (
    df["Ticket Subject"].iloc[0]
    + " "
    + df["Ticket Description"].iloc[0]
)

few_shot_prompt = f"""
You are a support ticket classifier.

Examples:

Ticket: App crashes while opening
Tag: Technical issue

Ticket: Payment charged twice
Tag: Billing inquiry

Ticket: Customer wants refund
Tag: Refund request

Ticket: Customer wants to cancel service
Tag: Cancellation request

Ticket: Customer asks product features
Tag: Product inquiry

Now classify this ticket.

Ticket:
{ticket}

Return top 3 most probable tags.
"""

print(few_shot_prompt)