import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# Page Config
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📊 Sales Forecasting Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload Sales Dataset", type=["csv"])

if uploaded_file is not None:

    # Load Data
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")

    st.success("File Uploaded Successfully ✅")

    # ================= KPI SECTION =================
    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Total Sales", f"{int(df['Sales'].sum()):,}")
    col2.metric("📊 Average Sales", f"{int(df['Sales'].mean()):,}")
    col3.metric("🚀 Highest Sales", f"{int(df['Sales'].max()):,}")

    st.markdown("---")

    # ================= HISTORICAL TREND =================
    fig = px.line(
        df,
        x="Date",
        y="Sales",
        title="Historical Sales Trend",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ================= FORECAST BUTTON =================
    if st.button("🔮 Run Forecast"):

        # Prepare data for Prophet
        prophet_df = df[['Date', 'Sales']].copy()
        prophet_df.columns = ['ds', 'y']

        # Log Transformation
        prophet_df['y'] = np.log(prophet_df['y'])

        # Tuned Prophet Model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        model.fit(prophet_df)

        # Future Dates (Next 6 Months)
        future = model.make_future_dataframe(periods=6, freq='MS')
        forecast = model.predict(future)

        # Reverse Log Transformation
        forecast['yhat'] = np.exp(forecast['yhat'])

        # ================= PLOT ACTUAL VS FORECAST =================
        fig2 = px.line(template="plotly_dark")

        fig2.add_scatter(
            x=df['Date'],
            y=df['Sales'],
            mode='lines',
            name="Actual"
        )

        fig2.add_scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name="Forecast"
        )

        fig2.update_layout(title="Actual vs Forecast Sales")

        st.plotly_chart(fig2, use_container_width=True)

        # ================= MAE CALCULATION =================
        predicted = forecast['yhat'][:len(df)]
        actual = df['Sales']

        mae = mean_absolute_error(actual, predicted)

        st.success(f"Improved Model MAE: {round(mae,2)}")

else:
    st.info("Please upload a CSV file to continue.")
