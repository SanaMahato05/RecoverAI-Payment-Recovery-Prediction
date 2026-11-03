import csv
import random
import os

# Create the data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

payment_methods = ["UPI", "Card", "Net Banking", "Wallet"]

failure_reasons = [
    "Bank Timeout",
    "Insufficient Funds",
    "Technical Error",
    "Card Expired",
    "Payment Declined"
]

with open("data/transactions.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "transaction_id",
        "customer_id",
        "amount",
        "payment_method",
        "failure_reason",
        "previous_failures",
        "recovery_status"
    ])

    for i in range(1, 1001):

        transaction_id = f"TX{i:05d}"
        customer_id = f"C{random.randint(1, 300):04d}"
        amount = random.randint(100, 10000)

        payment_method = random.choice(payment_methods)
        failure_reason = random.choice(failure_reasons)

        previous_failures = random.randint(0, 5)

        # Decide base recovery probability
        if failure_reason == "Bank Timeout":
            recovery_probability = 0.85

        elif failure_reason == "Technical Error":
            recovery_probability = 0.80

        elif failure_reason == "Payment Declined":
            recovery_probability = 0.50

        elif failure_reason == "Insufficient Funds":
            recovery_probability = 0.30

        elif failure_reason == "Card Expired":
            recovery_probability = 0.10

        # More previous failures reduce recovery probability
        recovery_probability -= previous_failures * 0.08

        # Keep probability between 0.05 and 0.95
        recovery_probability = max(
            0.05,
            min(0.95, recovery_probability)
        )

        # Decide whether the payment was actually recovered
        if random.random() < recovery_probability:
            recovery_status = "Recovered"
        else:
            recovery_status = "Not Recovered"

        writer.writerow([
            transaction_id,
            customer_id,
            amount,
            payment_method,
            failure_reason,
            previous_failures,
            recovery_status
        ])

print("Smart payment dataset created successfully!")