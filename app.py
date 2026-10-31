
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fraud Intelligence | FinTech",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #0B1120;
    color: #E2E8F0;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111B2E;
    border-right: 1px solid #263449;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] p {
    color: #E2E8F0;
}

/* Headings */
h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}

p, label {
    color: #CBD5E1;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #142238;
    border: 1px solid #263B56;
    padding: 20px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

/* Buttons */
.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #263B56;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #263449;
}

/* Custom section label */
.section-label {
    color: #38BDF8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 5px;
}

/* Page subtitle */
.subtitle {
    color: #94A3B8;
    font-size: 14px;
    margin-bottom: 25px;
}

/* Status */
.status {
    background: #142238;
    border: 1px solid #263B56;
    padding: 12px 16px;
    border-radius: 10px;
    color: #CBD5E1;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. DATA LOADING
# --------------------------------------------------

@st.cache_data
def generate_demo_data():

    np.random.seed(42)

    n = 10000

    fraud_count = 100

    classes = np.array(
        [0] * (n - fraud_count) + [1] * fraud_count
    )

    np.random.shuffle(classes)

    data = {
        "Time": np.sort(
            np.random.randint(0, 172800, n)
        ),
        "Amount": np.random.exponential(88, n),
        "Class": classes
    }

    for i in range(1, 29):

        values = np.random.normal(0, 1, n)

        # Introduce a difference in a few synthetic
        # features to make the demo charts informative.
        if i in [14, 17]:
            values[classes == 1] += np.random.normal(
                -3, 0.5, (classes == 1).sum()
            )

        data[f"V{i}"] = values

    return pd.DataFrame(data)


@st.cache_data
def load_uploaded_data(file_bytes):

    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data
def load_local_data(path):

    return pd.read_csv(path, nrows=50000)


def prepare_data(data):

    data = data.copy()

    required_columns = ["Time", "Amount", "Class"]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for col in required_columns:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data.dropna(
        subset=required_columns
    ).copy()

    data["Hour"] = (
        data["Time"] / 3600
    ) % 24

    data["Transaction Type"] = data["Class"].map(
        {0: "Legitimate", 1: "Fraud"}
    ).fillna("Unknown")

    return data


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

st.sidebar.markdown(
    """
    <div style="padding:10px 0 20px 0;">
        <h2 style="color:#38BDF8;margin-bottom:0;">
            🛡️ FRAUD INTEL
        </h2>
        <p style="color:#94A3B8;font-size:12px;">
            FINANCIAL SECURITY PLATFORM
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Transaction Analytics",
        "Fraud Analysis",
        "Model Performance"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.markdown("### Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload Credit Card CSV",
    type=["csv"]
)

local_path = st.sidebar.text_input(
    "Or enter local CSV path",
    placeholder="C:/data/creditcard.csv"
)

use_demo = st.sidebar.checkbox(
    "Use synthetic demo data",
    value=uploaded_file is None and not local_path.strip()
)

st.sidebar.caption(
    "Expected columns: Time, Amount, Class, V1–V28"
)

# --------------------------------------------------
# 5. LOAD SELECTED DATA
# --------------------------------------------------

try:

    if uploaded_file is not None:

        df = load_uploaded_data(
            uploaded_file.getvalue()
        )

        data_source = "Uploaded CSV"

    elif local_path.strip():

        df = load_local_data(
            local_path.strip()
        )

        data_source = "Local CSV"

    elif use_demo:

        df = generate_demo_data()

        data_source = "Synthetic Demo Data"

    else:

        st.info(
            "Upload a CSV file or enable synthetic demo data "
            "to open the dashboard."
        )

        st.stop()

    df = prepare_data(df)

except Exception as e:

    st.error(f"Unable to load dataset: {e}")
    st.stop()


# --------------------------------------------------
# 6. GLOBAL FILTERS
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.markdown("### Transaction Filters")

amount_min = float(df["Amount"].min())
amount_max = float(df["Amount"].max())

amount_range = st.sidebar.slider(
    "Transaction Amount",
    min_value=amount_min,
    max_value=amount_max,
    value=(amount_min, amount_max)
)

class_options = st.sidebar.multiselect(
    "Transaction Status",
    options=["Legitimate", "Fraud"],
    default=["Legitimate", "Fraud"]
)

filtered_df = df[
    (df["Amount"] >= amount_range[0]) &
    (df["Amount"] <= amount_range[1]) &
    (df["Transaction Type"].isin(class_options))
].copy()

st.sidebar.divider()

st.sidebar.caption(
    f"Data source: {data_source}"
)

st.sidebar.caption(
    f"Loaded records: {len(df):,}"
)

st.sidebar.caption(
    f"Filtered records: {len(filtered_df):,}"
)


# --------------------------------------------------
# 7. COMMON CALCULATIONS
# --------------------------------------------------

total_tx = len(filtered_df)

fraud_df = filtered_df[
    filtered_df["Class"] == 1
]

legit_df = filtered_df[
    filtered_df["Class"] == 0
]

fraud_count = len(fraud_df)
legit_count = len(legit_df)

fraud_pct = (
    fraud_count / total_tx * 100
    if total_tx > 0 else 0
)

avg_amount = (
    filtered_df["Amount"].mean()
    if total_tx > 0 else 0
)


# --------------------------------------------------
# 8. REUSABLE HEADER
# --------------------------------------------------

def page_header(title, description):

    st.markdown(
        '<p class="section-label">FRAUD INTELLIGENCE PLATFORM</p>',
        unsafe_allow_html=True
    )

    st.title(title)

    st.markdown(
        f'<p class="subtitle">{description}</p>',
        unsafe_allow_html=True
    )


def show_metrics():

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_tx:,}"
    )

    col2.metric(
        "Legitimate Transactions",
        f"{legit_count:,}"
    )

    col3.metric(
        "Fraudulent Transactions",
        f"{fraud_count:,}"
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_pct:.3f}%"
    )


# --------------------------------------------------
# 9. OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    page_header(
        "Fraud Monitoring Overview",
        "Monitor transaction activity, fraud patterns and financial risk."
    )

    st.markdown(
        f'<div class="status">● Data loaded successfully '
        f'&nbsp; | &nbsp; Source: {data_source}</div>',
        unsafe_allow_html=True
    )

    st.write("")

    show_metrics()

    st.write("")

    col1, col2 = st.columns([1.4, 1])

    with col1:

        st.subheader("Transaction Distribution")

        fig = px.pie(
            filtered_df,
            names="Transaction Type",
            hole=0.65,
            color="Transaction Type",
            color_discrete_map={
                "Legitimate": "#2563EB",
                "Fraud": "#EF4444",
                "Unknown": "#94A3B8"
            }
        )

        fig.update_traces(
            textinfo="percent+label",
            marker=dict(line=dict(
                color="#0B1120",
                width=3
            ))
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Transaction Summary")

        st.metric(
            "Average Transaction Amount",
            f"${avg_amount:,.2f}"
        )

        st.metric(
            "Total Fraud Amount",
            f"${fraud_df['Amount'].sum():,.2f}"
        )

        st.metric(
            "Legitimate Transaction Amount",
            f"${legit_df['Amount'].sum():,.2f}"
        )

        st.metric(
            "Fraud Transactions",
            f"{fraud_count:,}"
        )

    st.divider()

    st.subheader("Recent Transactions")

    st.dataframe(
        filtered_df.tail(10).sort_values(
            "Time", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 10. TRANSACTION ANALYTICS PAGE
# --------------------------------------------------

elif page == "Transaction Analytics":

    page_header(
        "Transaction Analytics",
        "Explore transaction volumes, amounts and time-based patterns."
    )

    show_metrics()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Transaction Amount Distribution")

        fig = px.histogram(
            filtered_df,
            x="Amount",
            color="Transaction Type",
            marginal="box",
            nbins=50,
            log_y=True,
            color_discrete_map={
                "Legitimate": "#2563EB",
                "Fraud": "#EF4444",
                "Unknown": "#94A3B8"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Hourly Transaction Trends")

        hourly = (
            filtered_df.groupby(
                ["Hour", "Transaction Type"]
            )
            .size()
            .reset_index(name="Transactions")
        )

        fig = px.bar(
            hourly,
            x="Hour",
            y="Transactions",
            color="Transaction Type",
            barmode="group",
            color_discrete_map={
                "Legitimate": "#2563EB",
                "Fraud": "#EF4444",
                "Unknown": "#94A3B8"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Transaction Amount Comparison")

    comparison = (
        filtered_df.groupby("Transaction Type")["Amount"]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
    )

    comparison.columns = [
        "Transaction Type",
        "Average Amount",
        "Median Amount",
        "Maximum Amount",
        "Transaction Count"
    ]

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Download Filtered Transactions")

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="filtered_transactions.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 11. FRAUD ANALYSIS PAGE
# --------------------------------------------------

elif page == "Fraud Analysis":

    page_header(
        "Fraud Pattern Analysis",
        "Investigate anomalous transaction behavior and feature patterns."
    )

    show_metrics()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("V14 Feature Distribution")

        if "V14" in filtered_df.columns:

            fig = px.box(
                filtered_df,
                x="Transaction Type",
                y="V14",
                color="Transaction Type",
                color_discrete_map={
                    "Legitimate": "#2563EB",
                    "Fraud": "#EF4444",
                    "Unknown": "#94A3B8"
                }
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("V14 feature is not available.")

    with col2:

        st.subheader("Feature Interaction: V17 vs V14")

        if {"V14", "V17"}.issubset(filtered_df.columns):

            sample = filtered_df.sample(
                min(5000, len(filtered_df)),
                random_state=42
            )

            fig = px.scatter(
                sample,
                x="V17",
                y="V14",
                color="Transaction Type",
                color_discrete_map={
                    "Legitimate": "#2563EB",
                    "Fraud": "#EF4444",
                    "Unknown": "#94A3B8"
                },
                opacity=0.7
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("V14 or V17 feature is not available.")

    st.divider()

    st.subheader("Fraud Transaction Records")

    if fraud_count > 0:

        st.dataframe(
            fraud_df.sort_values(
                "Amount", ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No fraudulent transactions found with the current filters."
        )

    st.divider()

    st.subheader("Feature Comparison")

    feature_columns = [
        col for col in filtered_df.columns
        if col.startswith("V") and col[1:].isdigit()
    ]

    if feature_columns and fraud_count > 0 and legit_count > 0:

        feature_summary = filtered_df.groupby(
            "Transaction Type"
        )[feature_columns].mean().T.reset_index()

        feature_summary.columns.name = None

        st.dataframe(
            feature_summary,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Feature comparison requires both transaction classes "
            "and available V1–V28 features."
        )


# --------------------------------------------------
# 12. MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    page_header(
        "Model Performance Evaluation",
        "Evaluate your trained fraud detection model using actual predictions."
    )

    st.info(
        "Upload a prediction CSV containing actual labels and model "
        "predictions. Required columns: y_true and y_pred. "
        "Optional column: y_score (fraud probability)."
    )

    prediction_file = st.file_uploader(
        "Upload Model Predictions CSV",
        type=["csv"],
        key="prediction_upload"
    )

    if prediction_file is not None:

        try:

            predictions = pd.read_csv(prediction_file)

            required = ["y_true", "y_pred"]

            if not all(
                col in predictions.columns for col in required
            ):

                st.error(
                    "Prediction CSV must contain y_true and y_pred."
                )

                st.stop()

            y_true = predictions["y_true"]
            y_pred = predictions["y_pred"]

            if (
                y_true.isna().any()
                or y_pred.isna().any()
                or not set(y_true.unique()).issubset({0, 1})
                or not set(y_pred.unique()).issubset({0, 1})
            ):

                st.error(
                    "y_true and y_pred must contain only binary "
                    "values 0 and 1, without missing values."
                )

                st.stop()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{accuracy_score(y_true, y_pred):.3f}"
            )

            col2.metric(
                "Precision",
                f"{precision_score(y_true, y_pred, zero_division=0):.3f}"
            )

            col3.metric(
                "Recall",
                f"{recall_score(y_true, y_pred, zero_division=0):.3f}"
            )

            col4.metric(
                "F1 Score",
                f"{f1_score(y_true, y_pred, zero_division=0):.3f}"
            )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1]
                )

                fig = px.imshow(
                    cm,
                    text_auto=True,
                    x=["Predicted Legit", "Predicted Fraud"],
                    y=["Actual Legit", "Actual Fraud"],
                    color_continuous_scale="Blues"
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with col2:

                st.subheader("Precision-Recall / ROC Curves")

                if "y_score" in predictions.columns:

                    y_score = predictions["y_score"]

                    if (
                        y_score.isna().any()
                        or not np.isfinite(y_score).all()
                        or not y_score.between(0, 1).all()
                    ):

                        st.warning(
                            "y_score must contain finite probabilities "
                            "between 0 and 1."
                        )

                    elif y_true.nunique() == 2:

                        precision, recall, _ = precision_recall_curve(
                            y_true,
                            y_score
                        )

                        fpr, tpr, _ = roc_curve(
                            y_true,
                            y_score
                        )

                        pr_auc = auc(recall[::-1], precision[::-1])
                        roc_auc = auc(fpr, tpr)

                        fig = go.Figure()

                        fig.add_trace(
                            go.Scatter(
                                x=recall,
                                y=precision,
                                name=f"PR Curve (AUC: {pr_auc:.3f})"
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=fpr,
                                y=tpr,
                                name=f"ROC Curve (AUC: {roc_auc:.3f})"
                            )
                        )

                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Recall / False Positive Rate",
                            yaxis_title="Precision / True Positive Rate"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Both classes are needed to calculate ROC and PR curves."
                        )

                else:

                    st.warning(
                        "Add a y_score column containing predicted "
                        "fraud probabilities to display PR and ROC curves."
                    )

        except Exception as e:

            st.error(
                f"Unable to evaluate model: {e}"
            )

    else:

        st.markdown(
            """
            ### What you can evaluate here

            - Accuracy: Overall prediction correctness.
            - Precision: How many predicted fraud cases were actually fraud.
            - Recall: How many actual fraud cases were detected.
            - F1 Score: Balance between precision and recall.
            - Confusion Matrix: Correct and incorrect predictions.
            - ROC and Precision-Recall curves: Model discrimination.
            """
        )

        st.warning(
            "Model performance is not displayed until actual prediction "
            "results are uploaded. This avoids presenting sample metrics "
            "as real model results."
        )


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#64748B;
                padding:15px;font-size:12px;">

        🛡️ FRAUD INTELLIGENCE DASHBOARD

        <br><br>

        Credit Card Fraud Monitoring & Analytics

        <br>

        Data-driven insights for financial security.

    </div>
    """,
    unsafe_allow_html=True
)
