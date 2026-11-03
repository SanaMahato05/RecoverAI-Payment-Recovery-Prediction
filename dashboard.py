
import streamlit as st
import joblib
import pandas as pd
import altair as alt
from pathlib import Path

# ==========================================
# PAGE SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="RecoverAI Dashboard",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(
        180deg, #EEF3F6 0%, #E6ECF0 55%, #DEE7EC 100%
    );
    color: #3B4252;
}

section[data-testid="stSidebar"] {
    background: #F6F7F9;
    border-right: 1px solid rgba(0,0,0,0.06);
}

.recoverai-header {
    padding: 1.6rem 2rem;
    border-radius: 18px;
    background: rgba(136,192,208,0.10);
    border: 1px solid rgba(136,192,208,0.28);
    margin-bottom: 1.4rem;
}

.recoverai-title {
    font-size: 3rem;
    font-weight: 800;
    color: #33526E;
    margin: 0;
}

.recoverai-subtitle {
    color: #64748B;
    font-size: 0.95rem;
    margin-top: 0.5rem;
}

.kpi-card {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}

.kpi-label {
    font-size: 0.8rem;
    color: #7B8496;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}

.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #2E3440;
}

.kpi-accent-total {
    border-left: 4px solid #6E93B7;
}

.kpi-accent-high {
    border-left: 4px solid #83A776;
}

.kpi-accent-medium {
    border-left: 4px solid #C9A648;
}

.kpi-accent-low {
    border-left: 4px solid #B4636C;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #2E3440;
    margin-bottom: 0.8rem;
}

hr, [data-testid="stDivider"] {
    border-color: rgba(0,0,0,0.08) !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(0,0,0,0.07);
}

.sidebar-title {
    font-weight: 700;
    font-size: 1.15rem;
    color: #2E3440;
}

.sidebar-sub {
    color: #7B8496;
    font-size: 0.82rem;
    margin-bottom: 1.2rem;
}

.filter-label {
    font-weight: 600;
    font-size: 0.85rem;
    color: #445069;
    margin-top: 0.9rem;
}

.filter-hint {
    color: #8B98AC;
    font-size: 0.72rem;
    margin-bottom: 0.3rem;
}

span[data-baseweb="tag"],
div[data-baseweb="tag"] {
    background-color: #DCE6EC !important;
    border: 1px solid rgba(110,147,183,0.40) !important;
}

span[data-baseweb="tag"] span,
div[data-baseweb="tag"] span {
    color: #3C5A76 !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED ML MODEL
# ==========================================

try:
    model = joblib.load(BASE_DIR / "recovery_model.pkl")
except Exception as e:
    st.error(f"Unable to load trained model: {e}")
    st.stop()


# ==========================================
# CATEGORY MAPPINGS
# ==========================================

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


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="recoverai-header">
    <p class="recoverai-title">
        💠 RecoverAI Recovery Dashboard
    </p>
    <p class="recoverai-subtitle">
        AI-prioritized recovery queue for failed payments —
        analyze, prioritize, and act in one place.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD PAYMENT DATA
# ==========================================

try:
    data = pd.read_csv(BASE_DIR / "data" / "transactions.csv")
except Exception as e:
    st.error(f"Unable to load transaction data: {e}")
    st.stop()

required_columns = [
    "transaction_id",
    "customer_id",
    "amount",
    "payment_method",
    "failure_reason",
    "previous_failures"
]

missing_columns = [
    col for col in required_columns if col not in data.columns
]

if missing_columns:
    st.error(f"Missing columns in dataset: {missing_columns}")
    st.stop()

if data.empty:
    st.warning("No transaction data available.")
    st.stop()

# Use first 20 payments
payments = data.head(20).copy()


# ==========================================
# PREPARE DATA FOR ML MODEL
# ==========================================

X = pd.DataFrame(index=payments.index)

X["amount"] = pd.to_numeric(
    payments["amount"], errors="coerce"
)

X["payment_method"] = (
    payments["payment_method"].map(payment_methods)
)

X["failure_reason"] = (
    payments["failure_reason"].map(failure_reasons)
)

X["previous_failures"] = pd.to_numeric(
    payments["previous_failures"], errors="coerce"
)

if X.isnull().any().any():
    st.error(
        "Some transaction values are missing or contain "
        "unrecognized payment methods or failure reasons. "
        "Please check your dataset."
    )
    st.stop()


# ==========================================
# PREDICT RECOVERY PROBABILITY
# ==========================================

try:
    probabilities = model.predict_proba(X)

    # Find the probability column corresponding to class 1
    positive_class_index = list(model.classes_).index(1)

    payments["recovery_probability"] = (
        probabilities[:, positive_class_index] * 100
    ).round(2)

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()


# ==========================================
# ASSIGN PRIORITY
# ==========================================

def get_priority(probability):

    if probability >= 70:
        return "HIGH"

    elif probability >= 40:
        return "MEDIUM"

    else:
        return "LOW"


payments["priority"] = (
    payments["recovery_probability"].apply(get_priority)
)


# ==========================================
# AI RECOVERY ACTION ENGINE
# ==========================================

def get_recovery_action(
    failure_reason,
    probability,
    previous_failures
):

    if previous_failures >= 5:
        return "Stop retries - Manual review"

    if failure_reason == "Bank Timeout":

        if probability >= 70:
            return "Retry automatically in 5 minutes"

        else:
            return "Retry after 15 minutes"

    elif failure_reason == "Technical Error":

        return "Check service and retry in 10 minutes"

    elif failure_reason == "Insufficient Funds":

        return "Notify customer to add funds"

    elif failure_reason == "Card Expired":

        return "Ask customer to update card"

    elif failure_reason == "Payment Declined":

        return "Suggest alternative payment method"

    return "Manual review"


# Add recommended recovery action

payments["recommended_action"] = payments.apply(
    lambda row: get_recovery_action(
        row["failure_reason"],
        row["recovery_probability"],
        row["previous_failures"]
    ),
    axis=1
)


# ==========================================
# SIDEBAR FILTERS
# ==========================================

all_priorities = ["HIGH", "MEDIUM", "LOW"]

all_reasons = sorted(
    payments["failure_reason"].unique().tolist()
)


def reset_filters():
    st.session_state["priority_filter"] = all_priorities
    st.session_state["reason_filter"] = all_reasons
    st.session_state["min_prob"] = 0
    st.session_state["search_id"] = ""


st.sidebar.markdown(
    '<p class="sidebar-title">🔎 Filter Queue</p>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<p class="sidebar-sub">Show only the payments you care about.</p>',
    unsafe_allow_html=True
)

st.sidebar.button(
    "↺ Reset all filters",
    use_container_width=True,
    on_click=reset_filters
)

st.sidebar.markdown(
    '<p class="filter-label">1. Priority level</p>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<p class="filter-hint">Which urgency tiers to include</p>',
    unsafe_allow_html=True
)

priority_filter = st.sidebar.pills(
    "Priority level",
    options=all_priorities,
    selection_mode="multi",
    default=all_priorities,
    key="priority_filter",
    label_visibility="collapsed"
)


st.sidebar.markdown(
    '<p class="filter-label">2. Failure reason</p>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<p class="filter-hint">Why the payment originally failed</p>',
    unsafe_allow_html=True
)

reason_filter = st.sidebar.multiselect(
    "Failure reason",
    options=all_reasons,
    default=all_reasons,
    key="reason_filter",
    label_visibility="collapsed"
)


st.sidebar.markdown(
    '<p class="filter-label">3. Minimum recovery chance</p>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<p class="filter-hint">Hide payments below this probability</p>',
    unsafe_allow_html=True
)

min_prob = st.sidebar.slider(
    "Minimum recovery chance",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
    key="min_prob",
    format="%d%%",
    label_visibility="collapsed"
)


st.sidebar.markdown(
    '<p class="filter-label">4. Search</p>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<p class="filter-hint">Match a Transaction ID or Customer ID</p>',
    unsafe_allow_html=True
)

search_id = st.sidebar.text_input(
    "Search",
    key="search_id",
    placeholder="e.g. TXN1042 or CUST88",
    label_visibility="collapsed"
)


# Apply filters

filtered = payments[
    payments["priority"].isin(
        priority_filter if priority_filter else all_priorities
    )
    & payments["failure_reason"].isin(
        reason_filter if reason_filter else all_reasons
    )
    & (payments["recovery_probability"] >= min_prob)
]

if search_id:
    filtered = filtered[
        filtered["transaction_id"].astype(str).str.contains(
            search_id, case=False, na=False
        )
        |
        filtered["customer_id"].astype(str).str.contains(
            search_id, case=False, na=False
        )
    ]

st.sidebar.divider()

st.sidebar.caption(
    f"📌 Showing {len(filtered)} of {len(payments)} payments"
)


# ==========================================
# SUMMARY METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

kpi_data = [
    (
        col1,
        "kpi-accent-total",
        "Total Failed Payments",
        len(payments)
    ),
    (
        col2,
        "kpi-accent-high",
        "High Priority",
        len(payments[payments["priority"] == "HIGH"])
    ),
    (
        col3,
        "kpi-accent-medium",
        "Medium Priority",
        len(payments[payments["priority"] == "MEDIUM"])
    ),
    (
        col4,
        "kpi-accent-low",
        "Low Priority",
        len(payments[payments["priority"] == "LOW"])
    )
]

for col, accent_class, label, value in kpi_data:

    with col:

        st.markdown(f"""
        <div class="kpi-card {accent_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PRIORITY DISTRIBUTION CHART
# ==========================================

st.divider()

st.markdown(
    '<p class="section-title">📊 Payment Priority Distribution</p>',
    unsafe_allow_html=True
)

priority_counts = (
    payments["priority"]
    .value_counts()
    .reindex(["HIGH", "MEDIUM", "LOW"])
    .fillna(0)
)

chart_col, spacer_col, info_col = st.columns([1.3, 0.2, 1])

priority_colors = {
    "HIGH": "#83A776",
    "MEDIUM": "#C9A648",
    "LOW": "#B4636C"
}

with chart_col:

    chart_df = priority_counts.rename("Count").reset_index()
    chart_df.columns = ["Priority", "Count"]

    bar_chart = (
        alt.Chart(chart_df)
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
            size=45
        )
        .encode(
            x=alt.X(
                "Priority:N",
                sort=["HIGH", "MEDIUM", "LOW"],
                title=None,
                axis=alt.Axis(
                    labelColor="#445069",
                    labelFontSize=13,
                    labelAngle=0
                )
            ),
            y=alt.Y(
                "Count:Q",
                title=None,
                axis=alt.Axis(
                    labelColor="#7B8496",
                    gridColor="#E5E9EF"
                )
            ),
            color=alt.Color(
                "Priority:N",
                scale=alt.Scale(
                    domain=list(priority_colors.keys()),
                    range=list(priority_colors.values())
                ),
                legend=None
            ),
            tooltip=["Priority", "Count"]
        )
        .properties(height=300, width=320)
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(
        bar_chart,
        use_container_width=True
    )


with info_col:

    st.markdown("##### Recovery Overview")

    st.markdown(f"""
    <div class="kpi-card kpi-accent-high"
         style="margin-bottom:0.7rem;">
        <div class="kpi-label">High Priority</div>
        <div class="kpi-value">{int(priority_counts["HIGH"])}</div>
    </div>

    <div class="kpi-card kpi-accent-medium"
         style="margin-bottom:0.7rem;">
        <div class="kpi-label">Medium Priority</div>
        <div class="kpi-value">{int(priority_counts["MEDIUM"])}</div>
    </div>

    <div class="kpi-card kpi-accent-low">
        <div class="kpi-label">Low Priority</div>
        <div class="kpi-value">{int(priority_counts["LOW"])}</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAYMENT RECOVERY QUEUE
# ==========================================

st.divider()

st.markdown(
    '<p class="section-title">📋 Payment Recovery Queue</p>',
    unsafe_allow_html=True
)

display_columns = [
    "transaction_id",
    "customer_id",
    "amount",
    "payment_method",
    "failure_reason",
    "previous_failures",
    "recovery_probability",
    "priority",
    "recommended_action"
]

if filtered.empty:

    st.info(
        "No payments match the current filters. "
        "Try adjusting the sidebar options."
    )

else:

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={

            "transaction_id": st.column_config.TextColumn(
                "Transaction ID"
            ),

            "customer_id": st.column_config.TextColumn(
                "Customer ID"
            ),

            "amount": st.column_config.NumberColumn(
                "Amount",
                format="₹ %.2f"
            ),

            "payment_method": st.column_config.TextColumn(
                "Method"
            ),

            "failure_reason": st.column_config.TextColumn(
                "Failure Reason"
            ),

            "previous_failures": st.column_config.NumberColumn(
                "Prev. Failures"
            ),

            "recovery_probability": st.column_config.ProgressColumn(
                "Recovery Probability",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),

            "priority": st.column_config.TextColumn(
                "Priority"
            ),

            "recommended_action": st.column_config.TextColumn(
                "Recommended Action",
                width="large"
            )
        }
    )

# ==========================================
# NEW TRANSACTION ANALYSIS
# ==========================================

st.divider()

st.markdown(
    '<p class="section-title">🆕 Check a New Transaction</p>',
    unsafe_allow_html=True
)

st.write(
    "Enter the details of a new failed payment to predict its recovery "
    "probability and get a recommended recovery action."
)

with st.form("new_transaction_form"):
    new_col1, new_col2 = st.columns(2)

    with new_col1:
        new_amount = st.number_input(
            "Payment Amount (₹)",
            min_value=1.0,
            value=2500.0,
            step=100.0,
            key="new_transaction_amount"
        )

        new_payment_method = st.selectbox(
            "Payment Method",
            options=list(payment_methods.keys()),
            key="new_transaction_method"
        )

    with new_col2:
        new_failure_reason = st.selectbox(
            "Failure Reason",
            options=list(failure_reasons.keys()),
            key="new_transaction_reason"
        )

        new_previous_failures = st.number_input(
            "Previous Failures",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            key="new_transaction_previous_failures"
        )

    analyze_new_transaction = st.form_submit_button(
        "🤖 Analyze New Transaction",
        use_container_width=True
    )


def get_new_transaction_strategy(failure_reason, probability, previous_failures):
    """Return a recovery action and customer-facing message."""

    # Repeated failures take priority over all other recovery strategies.
    if previous_failures >= 5:
        return {
            "action": "Stop repeated retries and send for manual review",
            "timing": "Escalate for manual review",
            "message": (
                "We noticed multiple unsuccessful payment attempts. "
                "To avoid repeated failures, automatic retries have been "
                "paused. Our team may need to review this payment."
            )
        }

    if failure_reason == "Bank Timeout":
        if probability >= 70:
            return {
                "action": "Retry the payment automatically",
                "timing": "Retry after 5 minutes",
                "message": (
                    "Your payment was interrupted by a temporary bank timeout. "
                    "We will try the payment again shortly."
                )
            }
        return {
            "action": "Retry after checking the bank response",
            "timing": "Retry after 15 minutes",
            "message": (
                "Your bank did not respond in time. Please wait while we "
                "check the payment status before trying again."
            )
        }

    if failure_reason == "Technical Error":
        return {
            "action": "Check the payment service and retry",
            "timing": "Retry after 10 minutes",
            "message": (
                "We experienced a temporary technical issue. "
                "Please wait while we check the service and retry your payment."
            )
        }

    if failure_reason == "Insufficient Funds":
        return {
            "action": "Ask the customer to add funds",
            "timing": "Retry after funds are available",
            "message": (
                "Your payment could not be completed due to insufficient funds. "
                "Please add funds and try again."
            )
        }

    if failure_reason == "Card Expired":
        return {
            "action": "Ask the customer to update card details",
            "timing": "Do not retry automatically",
            "message": (
                "Your card has expired. Please update your card details "
                "or use another payment method."
            )
        }

    if failure_reason == "Payment Declined":
        return {
            "action": "Suggest an alternative payment method",
            "timing": "Do not retry immediately",
            "message": (
                "Your payment was declined. Please try another payment method."
            )
        }

    return {
        "action": "Send payment for manual review",
        "timing": "Review before retrying",
        "message": "We are reviewing this failed payment before taking further action."
    }


if analyze_new_transaction:
    new_payment = pd.DataFrame([{
        "amount": new_amount,
        "payment_method": payment_methods[new_payment_method],
        "failure_reason": failure_reasons[new_failure_reason],
        "previous_failures": new_previous_failures
    }])

    try:
        new_prediction = model.predict(new_payment)[0]
        new_probabilities = model.predict_proba(new_payment)[0]
        positive_class_index = list(model.classes_).index(1)
        new_probability = float(
            new_probabilities[positive_class_index] * 100
        )

        new_priority = get_priority(new_probability)
        new_strategy = get_new_transaction_strategy(
            new_failure_reason,
            new_probability,
            new_previous_failures
        )

        st.divider()
        st.markdown("### 📊 New Transaction Recovery Analysis")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric("Recovery Probability", f"{new_probability:.2f}%")

        with result_col2:
            st.metric("Model Priority", new_priority)

        with result_col3:
            st.metric(
                "Previous Failures",
                int(new_previous_failures)
            )

        if new_prediction == 1:
            st.success("The model predicts a higher potential for payment recovery.")
        else:
            st.warning("The model predicts a lower potential for payment recovery.")

        st.markdown("### 🤖 Recovery Agent Recommendation")
        st.write(f"**Recommended Action:** {new_strategy['action']}")
        st.write(f"**When to Act:** {new_strategy['timing']}")

        st.markdown("### 💬 Suggested Customer Message")
        st.info(new_strategy["message"])

    except Exception as e:
        st.error(f"Unable to analyze this transaction: {e}")


st.caption("RecoverAI | ML-powered Payment Recovery System")