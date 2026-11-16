
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
BUSINESS = ROOT / "reports" / "business_insights"
MODELLING = ROOT / "reports" / "modelling"
NETWORK = ROOT / "reports" / "network_analysis"

st.set_page_config(
    page_title="AML Financial Crime Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.07);
    border: 1px solid rgba(128,128,128,0.14);
    border-radius: 12px;
    padding: 14px;
}
h1, h2, h3 {
    letter-spacing: -0.025em;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(path_string, limit=5000):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, nrows=limit)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(path_string):
    path = Path(path_string)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return {}


def find_column(df, candidates):
    lookup = {
        str(column).lower(): column
        for column in df.columns
    }
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def number(value, digits=0):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def percentage(value, digits=2):
    return f"{float(value) * 100:.{digits}f}%"


def report_table(path, title, description="", limit=5000):
    st.subheader(title)
    if description:
        st.caption(description)

    df = load_csv(str(path), limit)

    if df.empty:
        st.info(f"No data available in {path.name}.")
        return df

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download displayed CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=path.name,
        mime="text/csv",
        key=f"download_{path.stem}",
    )
    return df


def chart_from_report(df, title):
    if df.empty:
        return

    preferred_categories = [
        "payment_type", "transaction_type", "channel",
        "currency", "payment_currency", "laundering_type",
        "route", "date", "month", "category", "name",
    ]
    preferred_values = [
        "laundering_count", "laundering_transactions",
        "transaction_count", "total_transactions",
        "count", "transactions", "value",
    ]

    category = find_column(df, preferred_categories)
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()
    value = find_column(df, preferred_values)

    if category is None:
        other_columns = [
            col for col in df.columns
            if col not in numeric_columns
        ]
        category = other_columns[0] if other_columns else None

    if value is None and numeric_columns:
        value = numeric_columns[0]

    if category is None or value is None:
        return

    chart_data = df[[category, value]].copy()
    chart_data[value] = pd.to_numeric(
        chart_data[value], errors="coerce"
    )
    chart_data = chart_data.dropna(subset=[value])
    chart_data = chart_data.sort_values(
        value, ascending=False
    ).head(15)

    if chart_data.empty:
        return

    fig = px.bar(
        chart_data,
        x=value,
        y=category,
        orientation="h",
        title=title,
        color=value,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        height=420,
    )
    st.plotly_chart(fig, width="stretch")


def show_report_image(path):
    if path.exists():
        st.image(str(path), width="stretch")


def report_inventory(folder):
    if not folder.exists():
        return []
    return sorted(folder.glob("*.csv"))


def overview():
    st.title("🏦 AML Financial Crime Intelligence")
    st.caption(
        "Synthetic transaction monitoring | "
        "Historical financial crime analytics"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", "9,504,852")
    c2.metric("Labelled laundering", "9,873")
    c3.metric("Laundering rate", "0.104%")
    c4.metric("Data period", "Oct 2022 – Aug 2023")

    st.info(
        "These figures describe a synthetic dataset. "
        "A labelled transaction is not a verified real-world crime."
    )

    st.subheader("Project overview")
    st.write(
        "This project combines large-scale transaction analytics, "
        "supervised machine learning, and account-network analysis "
        "to investigate potential money laundering indicators."
    )

    metrics = load_json(str(MODELLING / "test_metrics.json"))
    if metrics:
        st.subheader("Machine learning summary")
        cols = st.columns(3)
        xgb = metrics.get("xgboost", {})
        cols[0].metric(
            "XGBoost PR-AUC",
            number(xgb.get("pr_auc"), 4),
        )
        cols[1].metric(
            "XGBoost precision",
            percentage(xgb.get("precision", 0)),
        )
        cols[2].metric(
            "XGBoost recall",
            percentage(xgb.get("recall", 0)),
        )

    st.subheader("Available business reports")
    files = report_inventory(BUSINESS)
    if not files:
        st.warning("Business insights CSV reports were not found.")
    else:
        for file in files:
            st.write(f"• {file.name}")

    st.subheader("Analysis methodology")
    st.write(
        "Transactions were cleaned in chunks, engineered using "
        "DuckDB, and split chronologically for model development. "
        "The held-out test period was evaluated without retuning "
        "the validation-selected thresholds."
    )


def transaction_risk():
    st.title("💳 Transaction Risk Analysis")
    st.caption(
        "Historical transaction patterns and labelled "
        "laundering activity"
    )

    files = report_inventory(BUSINESS)

    if not files:
        st.warning(
            "No business insights reports found in "
            "reports/business_insights/."
        )
        return

    st.write(
        "Explore transaction patterns by payment method, "
        "currency, location, typology, and time. "
        "Counts and amounts should not be interpreted as "
        "equivalent risk measures."
    )

    chosen = st.selectbox(
        "Select an analysis report",
        files,
        format_func=lambda path: (
            path.stem.replace("_", " ").title()
        ),
    )

    df = load_csv(str(chosen))
    chart_from_report(
        df,
        chosen.stem.replace("_", " ").title(),
    )

    report_table(
        chosen,
        "Underlying report",
        "The table contains the saved aggregate results.",
    )

    st.warning(
        "Amounts denominated in different currencies must "
        "not be added together without an exchange-rate "
        "conversion. Labelled laundering patterns are "
        "retrospective, not real-time predictions."
    )


def machine_learning():
    st.title("🤖 Machine Learning Performance")
    st.caption(
        "Chronological holdout evaluation | "
        "Random Forest vs XGBoost"
    )

    metrics = load_json(str(MODELLING / "test_metrics.json"))
    meta = load_json(str(MODELLING / "model_metadata.json"))

    if not metrics:
        st.error("test_metrics.json was not found.")
        return

    selected = meta.get("selected_model", "xgboost")
    st.success(
        f"Validation-selected model: "
        f"{selected.replace('_', ' ').title()}"
    )

    rows = []
    for name, values in metrics.items():
        rows.append({
            "Model": name.replace("_", " ").title(),
            "PR-AUC": values.get("pr_auc", 0),
            "ROC-AUC": values.get("roc_auc", 0),
            "Precision": values.get("precision", 0),
            "Recall": values.get("recall", 0),
            "F1": values.get("f1", 0),
            "True positives": values.get("true_positives", 0),
            "False positives": values.get("false_positives", 0),
            "False negatives": values.get("false_negatives", 0),
            "Threshold": values.get("threshold", 0),
        })

    comparison = pd.DataFrame(rows)

    st.subheader("Model comparison")
    st.dataframe(
        comparison.style.format({
            "PR-AUC": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1": "{:.2%}",
            "Threshold": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )

    long_data = comparison.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score",
    )
    long_data["Score"] *= 100

    fig = px.bar(
        long_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Precision, Recall and F1",
        labels={"Score": "Percentage (%)"},
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Confusion matrix")

    model_name = st.selectbox(
        "Choose model",
        list(metrics.keys()),
        format_func=lambda name: name.replace(
            "_", " "
        ).title(),
    )

    result = metrics[model_name]

    tn = result["true_negatives"]
    fp = result["false_positives"]
    fn = result["false_negatives"]
    tp = result["true_positives"]

    fig = go.Figure(
        data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted legitimate", "Predicted laundering"],
            y=["Actually legitimate", "Actually laundering"],
            text=[[f"{tn:,}", f"{fp:,}"],
                  [f"{fn:,}", f"{tp:,}"]],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        height=360,
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch")

    alerts = tp + fp
    st.subheader("Investigation workload")
    a, b, c = st.columns(3)
    a.metric("Alerts generated", f"{alerts:,}")
    b.metric("Correct alerts", f"{tp:,}")
    c.metric("False alerts", f"{fp:,}")

    st.write(
        f"The {model_name.replace('_', ' ').title()} model "
        f"generated {alerts:,} alerts, correctly identifying "
        f"{tp:,} labelled laundering transactions while "
        f"missing {fn:,}."
    )

    for filename in [
        "test_pr_auc_comparison.png",
        "test_precision_recall_curves.png",
    ]:
        show_report_image(MODELLING / filename)

    st.warning(
        "The models were trained with class rebalancing. "
        "Their scores are not calibrated laundering "
        "probabilities. Test thresholds were fixed using "
        "validation data, not tuned on the test set."
    )


def network_intelligence():
    st.title("🕸️ Account Network Intelligence")
    st.caption(
        "Account relationships and retrospective "
        "transaction-network patterns"
    )

    overview_path = NETWORK / "network_overview.csv"
    overview_df = load_csv(str(overview_path))

    if not overview_df.empty:
        st.subheader("Network overview")
        st.dataframe(
            overview_df,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Explore network findings")

    reports = {
        "Most connected accounts":
            "top_connected_accounts.csv",
        "Frequent transaction pairs":
            "top_transaction_pairs.csv",
        "Accounts with labelled laundering":
            "top_labelled_accounts.csv",
        "Reciprocal relationships":
            "reciprocal_pairs.csv",
        "Candidate three-account cycles":
            "candidate_three_account_cycles.csv",
    }

    selection = st.selectbox(
        "Network analysis",
        list(reports.keys()),
    )

    path = NETWORK / reports[selection]
    df = report_table(
        path,
        selection,
        "Retrospective structural relationships; "
        "not evidence of criminal activity.",
    )

    chart_from_report(df, selection)

    for path in sorted(NETWORK.glob("*.png")):
        st.subheader(path.stem.replace("_", " ").title())
        st.image(str(path), width="stretch")

    st.info(
        "Reciprocal transfers and circular account "
        "relationships can occur in legitimate activity. "
        "Candidate cycles are structural patterns and "
        "do not necessarily show time-ordered fund flows."
    )


def business_insights():
    st.title("📊 Business Insights")
    st.caption(
        "Evidence-based answers to financial crime "
        "monitoring questions"
    )

    metrics = load_json(str(MODELLING / "test_metrics.json"))
    xgb = metrics.get("xgboost", {})

    st.subheader("1. How common is labelled laundering?")
    st.write(
        "The synthetic dataset contains 9,873 labelled "
        "laundering transactions out of 9,504,852 total "
        "transactions, approximately 0.104%."
    )

    st.subheader("2. Can machine learning prioritise alerts?")
    if xgb:
        st.write(
            f"XGBoost achieved a test PR-AUC of "
            f"{xgb['pr_auc']:.4f} and precision of "
            f"{xgb['precision']:.2%}. "
            "This indicates useful ranking and alert "
            "prioritisation in the synthetic test data."
        )

    st.subheader("3. How many cases are missed?")
    if xgb:
        st.write(
            f"At the validation-selected threshold, "
            f"XGBoost detected "
            f"{xgb['true_positives']:,} labelled cases "
            f"and missed {xgb['false_negatives']:,}. "
            f"Recall was {xgb['recall']:.2%}."
        )

    st.subheader("4. What is the false-alert workload?")
    if xgb:
        alerts = (
            xgb["true_positives"]
            + xgb["false_positives"]
        )
        st.write(
            f"XGBoost generated {alerts:,} alerts, "
            f"including {xgb['false_positives']:,} "
            "false positives. This gives an estimate "
            "of the investigation workload at the "
            "selected threshold."
        )

    st.subheader(
        "5. Can account relationships support investigations?"
    )
    st.write(
        "The network reports identify highly connected "
        "accounts, frequent counterparties, reciprocal "
        "relationships, and candidate circular structures. "
        "These provide additional investigation context, "
        "but require further verification."
    )

    st.subheader("Operational recommendations")
    st.markdown("""
- Use model scores to prioritise transaction reviews,
  not to automatically determine criminal activity.
- Monitor precision, recall, alert volume and missed cases.
- Combine transaction-level alerts with account-network
  evidence and analyst investigation.
- Review class imbalance, score calibration and
  synthetic-data limitations before any real-world use.
- Validate future changes on new chronological data,
  rather than repeatedly tuning against the test set.
""")

    st.warning(
        "All findings are based on synthetic historical "
        "transactions. No production effectiveness "
        "or regulatory compliance is claimed."
    )


PAGES = {
    "Overview": overview,
    "Transaction Risk": transaction_risk,
    "Machine Learning": machine_learning,
    "Network Intelligence": network_intelligence,
    "Business Insights": business_insights,
}

with st.sidebar:
    st.title("🏦 AML Intelligence")
    st.caption("Financial Crime Analytics")
    st.divider()

    page = st.radio(
        "Navigation",
        list(PAGES.keys()),
    )

    st.divider()
    st.caption("Dataset: Synthetic AML transactions")
    st.caption("Records: 9,504,852")
    st.caption("Model: XGBoost / Random Forest")
    st.caption("For research and portfolio use")

PAGES[page]()
