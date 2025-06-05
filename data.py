import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import base64
import io

# --- واجهة المستخدم ---
st.set_page_config(page_title="GDP Forecasting", layout="wide")

st.title("🌍 GDP Forecasting App")
st.markdown("This app allows you to visualize and forecast GDP data using Linear Regression.")

# --- تحميل البيانات من المستخدم ---
uploaded_file = st.file_uploader("Upload your GDP data file (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df)

    # التحقق من الأعمدة
    if "Year" in df.columns and "GDP" in df.columns:
        # تحويل السنة إلى عدد صحيح
        df["Year"] = df["Year"].astype(int)

        # --- رسم بياني للناتج المحلي ---
        fig = px.line(df, x="Year", y="GDP", title="GDP Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # --- التدريب والتنبؤ ---
        future_years = st.slider("Select number of years to forecast", 1, 20, 5)

        X = df[["Year"]]
        y = df["GDP"]
        model = LinearRegression()
        model.fit(X, y)

        future = pd.DataFrame({"Year": range(df["Year"].max() + 1, df["Year"].max() + future_years + 1)})
        future["GDP_Forecast"] = model.predict(future[["Year"]])

        # --- عرض النتائج ---
        st.subheader("📈 Forecasted Data")
        st.dataframe(future)

        # --- رسم بياني للتنبؤ ---
        combined = pd.concat([df[["Year", "GDP"]].rename(columns={"GDP": "Actual"}), 
                              future.rename(columns={"GDP_Forecast": "Forecast"})],
                             ignore_index=True)

        fig2 = px.line(combined, x="Year", y=combined.columns[1],
                       title="Actual and Forecasted GDP", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

        # --- تحميل النتائج ---
        def get_table_download_link(df, filename="forecast.csv"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Forecast as CSV</a>'

        st.markdown(get_table_download_link(future), unsafe_allow_html=True)

    else:
        st.error("Your CSV file must contain 'Year' and 'GDP' columns.")
else:
    st.info("Please upload a CSV file to proceed.")
