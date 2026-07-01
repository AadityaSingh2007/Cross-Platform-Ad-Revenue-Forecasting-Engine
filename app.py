import streamlit as st
import joblib
import pandas as pd
from ai_insights import generate_insights
from feature_engineering import create_features

model = joblib.load("best_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Revenue Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 1.5rem;
    }

    /* Card container for metrics */
    div[data-testid="stMetric"] {
        background-color: #1c1f26;
        border: 1px solid #2d313c;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #9aa3af !important;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        color: #f5f5f7 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Section headers */
    h1 {
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #2d313c;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1rem;
        padding: 4px 0;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #6366f1;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #4f46e5;
        color: white;
    }

    /* Success box */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Summary card */
    .summary-card {
        background-color: #1c1f26;
        border: 1px solid #2d313c;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 1rem;
    }
    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #2a2e38;
        font-size: 0.95rem;
    }
    .summary-row:last-child {
        border-bottom: none;
    }
    .summary-label {
        color: #9aa3af;
    }
    .summary-value {
        color: #f5f5f7;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title(" Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Manual Prediction",
        "Batch Prediction",
        "Model Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Revenue Forecasting · v1.0")

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "Home":

    st.title("Revenue Forecasting Dashboard")
    st.caption("Welcome to the Revenue Forecasting System")

    st.markdown("#### Overview")
    st.write(
        "This application predicts campaign revenue using a trained Random Forest model."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Best Model", "Random Forest")

    with col2:
        st.metric("R² Score", "0.845")

    with col3:
        st.metric("MAE", "227")

    with col4:
        st.metric("RMSE", "557")

    st.divider()

    st.markdown("#### 📂 Dataset")
    st.info("Google Ads + Bing Ads Campaign Data")

elif page == "Manual Prediction":

    st.title("Manual Revenue Prediction")
    st.caption("Enter campaign details to predict expected revenue.")

    st.markdown("##### Campaign Info")
    top1, top2 = st.columns(2)
    with top1:
        date = st.date_input("Campaign Date")
    with top2:
        campaign_name = st.text_input("Campaign Name", placeholder="e.g. Summer Sale 2026")

    st.markdown("##### Campaign Metrics")
    col1, col2 = st.columns(2)

    with col1:

        campaign_budget = st.number_input(
            "Campaign Budget ($)",
            min_value=0.0,
            value=100.0
        )

        spend = st.number_input(
            "Spend ($)",
            min_value=0.0,
            value=50.0
        )

        impressions = st.number_input(
            "Impressions",
            min_value=0,
            value=1000
        )

    with col2:

        clicks = st.number_input(
            "Clicks",
            min_value=0,
            value=150
        )

        campaign_type = st.selectbox(
            "Campaign Type",
            [
                "SEARCH",
                "DISPLAY",
                "SHOPPING",
                "VIDEO",
                "DEMAND_GEN",
                "PERFORMANCEMAX"
            ]
        )

        source = st.selectbox(
            "Platform",
            [
                "Google",
                "Bing"
            ]
        )

    st.divider()

    predict_clicked = st.button("Predict Revenue", use_container_width=False)

    if predict_clicked:
        input_data = {
            "Date": str(date),
            "CampaignName": campaign_name,
            "Conversions": 0,
            "campaign_budget": campaign_budget,
            "Spend": spend,
            "Impressions": impressions,
            "Clicks": clicks,
            "CampaignType": campaign_type,
            "Source": source
        }

        input_df = pd.DataFrame([input_data])

        input_df = create_features(input_df)

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = model.predict(input_df)[0]
        insights = generate_insights(
            campaign_budget,
            spend,
            impressions,
            clicks,
            campaign_type,
            source,
            prediction
            )
        st.markdown("### Campaign Summary")

        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-row"><span class="summary-label">Campaign Budget</span><span class="summary-value">${campaign_budget}</span></div>
            <div class="summary-row"><span class="summary-label">Spend</span><span class="summary-value">${spend}</span></div>
            <div class="summary-row"><span class="summary-label">Impressions</span><span class="summary-value">{impressions}</span></div>
            <div class="summary-row"><span class="summary-label">Clicks</span><span class="summary-value">{clicks}</span></div>
            <div class="summary-row"><span class="summary-label">Campaign Type</span><span class="summary-value">{campaign_type}</span></div>
            <div class="summary-row"><span class="summary-label">Platform</span><span class="summary-value">{source}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.success("Revenue Predicted Successfully!")

        st.metric(
            "Predicted Revenue",
            f"${prediction:,.2f}"
        )
        st.divider()
        st.subheader("🤖 AI Business Insights")
        st.markdown(insights)

elif page == " Batch Prediction":

    st.title(" Batch Prediction")
    st.caption("Upload a CSV file and predict revenue for all campaigns.")

    uploaded_file = st.file_uploader(
        "Upload Campaign CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.dataframe(df.head())

        if st.button("🚀 Predict All Campaigns"):

            df_features = create_features(df.copy())

            df_features = df_features.reindex(
                columns=feature_columns,
                fill_value=0
            )

            predictions = model.predict(df_features)

            df["Predicted Revenue"] = predictions

            st.success("Prediction Completed!")

            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Predictions",
                csv,
                "predicted_campaigns.csv",
                "text/csv"
            )

elif page == "Model Insights":

    st.title("Model Insights")

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R²", "0.845")

    with col2:
        st.metric("MAE", "227")

    with col3:
        st.metric("RMSE", "557")

    st.divider()

    st.subheader("Feature Importance")

    feature_importance = pd.read_csv("feature_importance.csv")

    st.dataframe(feature_importance)

    st.bar_chart(
        feature_importance.set_index("Feature")
    )

    st.divider()

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({

        "Model":[
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ],

        "R²":[
            0.7596,
            0.8454,
            0.7339
        ],

        "MAE":[
            289.72,
            227.66,
            236.64
        ],

        "RMSE":[
            695.07,
            557.31,
            731.30
        ]

    })
    st.dataframe(comparison)
    st.divider()

    st.subheader("Actual vs Predicted")

    st.image(
        "actual_vs_predicted.png",
         use_container_width=True
    )

    st.subheader("Residual Plot")

    st.image(
        "residual_plot.png",
        use_container_width=True
    )
