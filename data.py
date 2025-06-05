import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import io
from datetime import datetime

# واجهة المستخدم
st.set_page_config(page_title="GDP Forecasting App", layout="wide")

st.title("📊 GDP Analysis and Forecasting App")

# تحميل البيانات
uploaded_file = st.file_uploader("Upload your GDP data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # التحقق من الأعمدة
    if 'Year' not in df.columns or 'GDP' not in df.columns:
        st.error("CSV file must contain 'Year' and 'GDP' columns.")
    else:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
        df = df.dropna()

        st.subheader("📈 Historical GDP Data")
        st.dataframe(df)

        # رسم البيانات الأصلية
        fig = px.line(df, x="Year", y="GDP", title="GDP Over the Years")
        st.plotly_chart(fig, use_container_width=True)

        # نموذج الانحدار الخطي للتنبؤ
        forecast_years = st.slider("📅 How many future years to forecast?", 1, 10, 5)

        X = df[["Year"]]
        y = df["GDP"]

        model = LinearRegression()
        model.fit(X, y)

        future_years = pd.DataFrame({
            "Year": list(range(df["Year"].max() + 1, df["Year"].max() + forecast_years + 1))
        })

        future_preds = model.predict(future_years)

        forecast_df = pd.concat([
            df,
            pd.DataFrame({
                "Year": future_years["Year"],
                "GDP": future_preds
            })
        ])

        st.subheader("📉 Forecasted GDP")
        fig2 = px.line(forecast_df, x="Year", y="GDP", title="Forecasted GDP")
        st.plotly_chart(fig2, use_container_width=True)

        # تحميل التوقعات
        buffer = io.BytesIO()
        forecast_df.to_csv(buffer, index=False)
        st.download_button(
            label="📥 Download Forecast Data as CSV",
            data=buffer.getvalue(),
            file_name=f"gdp_forecast_{datetime.now().year}.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a CSV file with 'Year' and 'GDP' columns to begin.")
