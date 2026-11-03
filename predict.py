import joblib
import pandas as pd

# Load the saved model
model = joblib.load("recovery_model.pkl")

payment_methods = {
    "card": 0,
    "net banking": 1,
    "upi": 2,
    "wallet": 3
}

failure_reasons = {
    "bank timeout": 0,
    "card expired": 1,
    "insufficient funds": 2,
    "payment declined": 3,
    "technical error": 4
}

# Get input from user
amount = float(input("Enter payment amount: ₹"))

print("\nAvailable payment methods:")
print("Card, Net Banking, UPI, Wallet")
payment_method = input("Enter payment method: ").strip().lower()

print("\nAvailable failure reasons:")
print("Bank Timeout, Card Expired, Insufficient Funds, Payment Declined, Technical Error")
failure_reason = input("Enter failure reason: ").strip().lower()

previous_failures = int(
    input("\nEnter number of previous failures: ")
)

# Validate user input
if payment_method not in payment_methods:
    print("Invalid payment method!")
    exit()

if failure_reason not in failure_reasons:
    print("Invalid failure reason!")
    exit()

# Create input for the model
new_payment = pd.DataFrame([{
    "amount": amount,
    "payment_method": payment_methods[payment_method],
    "failure_reason": failure_reasons[failure_reason],
    "previous_failures": previous_failures
}])

# Predict
prediction = model.predict(new_payment)[0]

# Recovery probability
probability = model.predict_proba(new_payment)[0][1] * 100

# Display result
print("\n--- RECOVERAI PREDICTION ---")
print(f"Amount: ₹{amount}")
print(f"Payment Method: {payment_method.upper()}")
print(f"Failure Reason: {failure_reason.title()}")
print(f"Previous Failures: {previous_failures}")

print(f"\nRecovery Probability: {probability:.2f}%")

if prediction == 1:
    print("Recommendation: Attempt to recover this payment")
else:
    print("Recommendation: Do not prioritize recovery")