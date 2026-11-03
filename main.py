import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="RecoverAI",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# LOAD ML MODEL
# -----------------------------
model = joblib.load("recovery_model.pkl")

# Category mappings
payment_methods = {
    "Card": 0,
    "Net Banking": 1,
    "UPI": 2,
    "Wallet": 3
}

failure_reasons = {
    "Bank Timeout": 0,
    "Card Expired": 1,
    "Insufficient Funds": 2,
    "Payment Declined": 3,
    "Technical Error": 4
}


# -----------------------------
# AI RECOVERY STRATEGY ENGINE
# -----------------------------
def get_recovery_strategy(
        failure_reason,
        probability,
        previous_failures):

    # Bank Timeout
    if failure_reason == "Bank Timeout":
        if probability >= 70:
            return {
                "priority": "HIGH",
                "action": "Retry the payment automatically",
                "timing": "Retry after 5 minutes",
                "message": "Your payment is being retried due to a temporary bank timeout."
            }

    # Technical Error
    elif failure_reason == "Technical Error":
        return {
            "priority": "MEDIUM",
            "action": "Retry after checking payment service",
            "timing": "Retry after 10 minutes",
            "message": "We experienced a temporary technical issue. Please wait while we retry your payment."
        }

    # Insufficient Funds
    elif failure_reason == "Insufficient Funds":
        return {
            "priority": "LOW",
            "action": "Do not automatically retry repeatedly",
            "timing": "Ask customer to add funds first",
            "message": "Your payment could not be completed due to insufficient funds. Please add funds and try again."
        }

    # Card Expired
    elif failure_reason == "Card Expired":
        return {
            "priority": "LOW",
            "action": "Ask customer to update card details",
            "timing": "Do not retry automatically",
            "message": "Your card has expired. Please update your card details or use another payment method."
        }

    # Payment Declined
    elif failure_reason == "Payment Declined":
        return {
            "priority": "MEDIUM",
            "action": "Suggest an alternative payment method",
            "timing": "Do not retry immediately",
            "message": "Your payment was declined. Please try another payment method."
        }

    # High number of failures
    if previous_failures >= 5:
        return {
            "priority": "LOW",
            "action": "Stop repeated retries",
            "timing": "Escalate for manual review",
            "message": "Multiple payment failures detected. Further automatic retries have been stopped."
        }

    # Default
    if probability >= 70:
        return {
            "priority": "HIGH",
            "action": "Attempt payment recovery",
            "timing": "Retry automatically",
            "message": "We are attempting to recover your failed payment."
        }
    else:
        return {
            "priority": "LOW",
            "action": "Do not prioritize recovery",
            "timing": "Manual review recommended",
            "message": "This payment has a low probability of recovery."
        }


# -----------------------------
# USER INTERFACE
# -----------------------------
st.title("🤖 RecoverAI")
st.subheader("AI-Powered Payment Recovery Prediction")

st.write(
    "Enter the failed payment details. RecoverAI will predict "
    "the recovery probability and recommend the next best action."
)

# User inputs
amount = st.number_input(
    "Payment Amount (₹)",
    min_value=1.0,
    value=2500.0
)

payment_method = st.selectbox(
    "Payment Method",
    list(payment_methods.keys())
)

failure_reason = st.selectbox(
    "Failure Reason",
    list(failure_reasons.keys())
)

previous_failures = st.number_input(
    "Previous Failures",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)


# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🤖 Analyze & Recover"):

    new_payment = pd.DataFrame([{
        "amount": amount,
        "payment_method": payment_methods[payment_method],
        "failure_reason": failure_reasons[failure_reason],
        "previous_failures": previous_failures
    }])

    # ML prediction
    prediction = model.predict(new_payment)[0]

    probability = (
        model.predict_proba(new_payment)[0][1] * 100
    )

    # Get AI strategy
    strategy = get_recovery_strategy(
        failure_reason,
        probability,
        previous_failures
    )

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.divider()
    st.header("📊 Recovery Analysis")

    st.metric(
        "Recovery Probability",
        f"{probability:.2f}%"
    )

    if prediction == 1:
        st.success("High potential for payment recovery")
    else:
        st.warning("Low potential for payment recovery")

    # -----------------------------
    # AI AGENT DECISION
    # -----------------------------
    st.divider()
    st.header("🤖 AI Recovery Agent Decision")

    st.write(f"**Priority:** {strategy['priority']}")
    st.write(f"**Recommended Action:** {strategy['action']}")
    st.write(f"**When to Act:** {strategy['timing']}")

    st.divider()
    st.subheader("💬 Suggested Customer Message")

    st.info(strategy["message"])