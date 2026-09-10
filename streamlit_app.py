import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# NORTHBAY FORESIGHT - STREAMLIT FRONTEND
# ============================================================

st.set_page_config(
    page_title="NorthBay FORESIGHT",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = st.secrets.get(
    "API_URL",
    "https://ai-powered-inventory-inventory-analysis.onrender.com"
).rstrip("/")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 0;
}

.dashboard-subtitle {
    color: #64748b;
    font-size: 16px;
    margin-bottom: 25px;
}

.kpi-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.kpi-title {
    color: #64748b;
    font-size: 14px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
    margin-top: 6px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background: #eef6ff;
    border-left: 5px solid #2563eb;
}

.warning-box {
    padding: 15px;
    border-radius: 10px;
    background: #fff7ed;
    border-left: 5px solid #f97316;
}

.danger-box {
    padding: 15px;
    border-radius: 10px;
    background: #fef2f2;
    border-left: 5px solid #dc2626;
}

.success-box {
    padding: 15px;
    border-radius: 10px;
    background: #f0fdf4;
    border-left: 5px solid #16a34a;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):
    try:
        value = float(value or 0)
        return f"₹{value:,.0f}"
    except:
        return "₹0"


def api_get(endpoint, params=None, timeout=60):

    try:

        url = f"{API_URL}{endpoint}"

        response = requests.get(
            url,
            params=params,
            timeout=timeout
        )

        response.raise_for_status()

        if not response.text.strip():
            st.error(
                f"API returned an empty response: {endpoint}"
            )
            return {}

        return response.json()

    except requests.exceptions.Timeout:

        st.error(
            f"API timeout while requesting {endpoint}"
        )

        return {}

    except requests.exceptions.RequestException as e:

        st.error(
            f"API connection error: {e}"
        )

        return {}

    except ValueError:

        st.error(
            f"API returned invalid JSON: {endpoint}"
        )

        return {}


def dataframe_from_response(response):

    if not response:
        return pd.DataFrame()

    data = response.get("data", [])

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def safe_number(value):

    try:
        return float(value)
    except:
        return 0.0


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">📦 NORTHBAY FORESIGHT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'AI-powered Demand Forecasting & Inventory Decision Intelligence'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# API STATUS
# ============================================================

health = api_get("/health")

if health:

    st.sidebar.success("🟢 Flask API Connected")

else:

    st.sidebar.error("🔴 Flask API Unavailable")


st.sidebar.markdown("### System")

st.sidebar.write(
    f"API: `{API_URL}`"
)

st.sidebar.caption(
    "Streamlit acts as the analytics frontend. "
    "Forecast and inventory calculations are served by Flask."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("## 🔎 Filters")

sku_id = st.sidebar.text_input(
    "SKU ID",
    placeholder="Example: SKU00001"
)

store_id = st.sidebar.text_input(
    "Store ID",
    placeholder="Optional"
)

category = st.sidebar.text_input(
    "Category",
    placeholder="Optional"
)

risk_type = st.sidebar.selectbox(
    "Risk Type",
    [
        "All",
        "STOCKOUT",
        "OVERSTOCK",
        "NORMAL"
    ]
)

limit = st.sidebar.slider(
    "Records to display",
    min_value=20,
    max_value=500,
    value=100,
    step=20
)


# ============================================================
# BUILD PARAMETERS
# ============================================================

params = {
    "limit": limit
}

if sku_id.strip():
    params["sku_id"] = sku_id.strip()

if store_id.strip():
    params["store_id"] = store_id.strip()

if category.strip():
    params["category"] = category.strip()

if risk_type != "All":
    params["risk_type"] = risk_type


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("Loading NorthBay FORESIGHT data..."):

    insights = api_get("/insights")

    forecast_response = api_get(
        "/forecast",
        params=params
    )

    risk_response = api_get(
        "/risk",
        params=params
    )

    reorder_response = api_get(
        "/reorder",
        params=params
    )

    markdown_response = api_get(
        "/markdown",
        params=params
    )


forecast_df = dataframe_from_response(
    forecast_response
)

risk_df = dataframe_from_response(
    risk_response
)

reorder_df = dataframe_from_response(
    reorder_response
)

markdown_df = dataframe_from_response(
    markdown_response
)


# ============================================================
# KPI SECTION
# ============================================================

st.markdown("## 📊 Executive Overview")

c1, c2, c3, c4 = st.columns(4)

forecast_count = safe_number(
    insights.get("forecast_records", 0)
)

stockout_count = safe_number(
    insights.get("stockout_count", 0)
)

overstock_count = safe_number(
    insights.get("overstock_count", 0)
)

value_at_stake = safe_number(
    insights.get("total_value_at_stake", 0)
)

with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Forecast Records</div>
            <div class="kpi-value">
                {forecast_count:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">🔴 Stockout Risk</div>
            <div class="kpi-value">
                {stockout_count:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">🟠 Overstock Risk</div>
            <div class="kpi-value">
                {overstock_count:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">💰 Value at Stake</div>
            <div class="kpi-value">
                {money(value_at_stake)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# BUSINESS DECISION
# ============================================================

st.markdown("## 🚨 Priority Decision")

if stockout_count > 0:

    st.markdown(
        f"""
        <div class="danger-box">
        <b>REORDER REQUIRED</b><br>
        {stockout_count:,.0f} inventory records indicate
        potential stockout risk.
        </div>
        """,
        unsafe_allow_html=True
    )

elif overstock_count > 0:

    st.markdown(
        f"""
        <div class="warning-box">
        <b>SELL / MARKDOWN</b><br>
        {overstock_count:,.0f} records indicate excess inventory.
        Consider promotions, markdowns or accelerated sales.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="success-box">
        <b>NO IMMEDIATE ACTION</b><br>
        Current inventory risk indicators are stable.
        Continue monitoring demand and inventory.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RISK CHARTS
# ============================================================

st.markdown("## 📈 Inventory Risk Analysis")

col1, col2 = st.columns(2)

with col1:

    risk_counts = insights.get(
        "risk_counts",
        {}
    )

    if risk_counts:

        risk_chart_df = pd.DataFrame(
            {
                "Risk": list(risk_counts.keys()),
                "Count": list(risk_counts.values())
            }
        )

        fig = px.pie(
            risk_chart_df,
            names="Risk",
            values="Count",
            hole=0.45,
            title="Risk Distribution"
        )

        fig.update_layout(
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Risk distribution data unavailable."
        )


with col2:

    value_df = pd.DataFrame(
        {
            "Risk": [
                "Stockout",
                "Overstock"
            ],
            "Value": [
                safe_number(
                    insights.get(
                        "stockout_value",
                        0
                    )
                ),
                safe_number(
                    insights.get(
                        "overstock_value",
                        0
                    )
                )
            ]
        }
    )

    fig = px.bar(
        value_df,
        x="Risk",
        y="Value",
        title="Financial Value at Stake",
        text_auto=True
    )

    fig.update_layout(
        height=400,
        yaxis_title="₹"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FORECAST
# ============================================================

st.markdown("## 📈 Demand Forecast")

if not forecast_df.empty:

    date_col = None

    for c in ["date", "Date", "forecast_date"]:

        if c in forecast_df.columns:

            date_col = c
            break

    prediction_col = None

    for c in [
        "prediction",
        "predicted_demand",
        "forecast",
        "y_pred"
    ]:

        if c in forecast_df.columns:

            prediction_col = c
            break

    if date_col and prediction_col:

        temp = forecast_df.copy()

        temp[date_col] = pd.to_datetime(
            temp[date_col],
            errors="coerce"
        )

        temp[prediction_col] = pd.to_numeric(
            temp[prediction_col],
            errors="coerce"
        )

        temp = temp.dropna(
            subset=[
                date_col,
                prediction_col
            ]
        )

        if not temp.empty:

            fig = px.line(
                temp,
                x=date_col,
                y=prediction_col,
                markers=True,
                title="Predicted Demand Over Time"
            )

            fig.update_layout(
                height=450,
                xaxis_title="Date",
                yaxis_title="Predicted Units"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Forecast records exist but "
                "valid date/prediction values were not found."
            )

    else:

        st.info(
            "Forecast API is connected, but the expected "
            "date/prediction columns are unavailable."
        )

else:

    st.warning(
        "No forecast data available for the selected filters."
    )


# ============================================================
# REORDER
# ============================================================

st.markdown("## 🔴 Reorder Priority")

if not reorder_df.empty:

    st.dataframe(
        reorder_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Business action: prioritize purchasing/replenishment "
        "for high-risk items."
    )

else:

    st.success(
        "No reorder records found for the selected filters."
    )


# ============================================================
# MARKDOWN
# ============================================================

st.markdown("## 🟠 Markdown / Sell Now")

if not markdown_df.empty:

    st.dataframe(
        markdown_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Business action: accelerate sales using promotions, "
        "markdowns or inventory movement."
    )

else:

    st.info(
        "No markdown-priority records found."
    )


# ============================================================
# RISK DETAILS
# ============================================================

st.markdown("## 🚨 Risk Analysis")

if not risk_df.empty:

    st.dataframe(
        risk_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No risk records available for the selected filters."
    )


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.markdown("## 💡 Business Interpretation")

st.markdown(
"""
<div class="info-box">

<b>REORDER</b> → Inventory may run out soon.  
Purchase/replenish the product before the expected stockout.

<br><br>

<b>MARKDOWN / SELL NOW</b> → Inventory is higher than expected demand.  
Increase sales velocity through promotions, discounts or redistribution.

<br><br>

<b>WATCH</b> → Risk is not immediately critical but should be monitored.

<br><br>

<b>NO ACTION</b> → Inventory currently appears stable.

<br><br>

<b>RED FLAG</b> → Immediate management attention is recommended.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "NorthBay FORESIGHT | AI-powered Inventory Decision Intelligence"
)
