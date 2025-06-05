import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# إعداد الصفحة
st.set_page_config(page_title="GDP Forecasting App", layout="wide")

# عنوان التطبيق
st.title("🌍 GDP Analysis and Forecasting App")

# تحميل البيانات من المستخدم
uploaded_file = st.file_uploader("📤 Upload your GDP data (CSV file)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # التحقق من الأعمدة
    if 'Year' in df.columns and 'GDP' in df.columns:
        st.success("✅ Data loaded successfully!")

        # عرض البيانات الأصلية
        st.subheader("📊 Original Data")
        st.dataframe(df)

        # رسم البيانات الأصلية
        fig = px.line(df, x='Year', y='GDP', title='GDP Over Time')
        st.plotly_chart(fig, use_container_width=True)

        # إعداد النموذج
        X = df['Year'].values.reshape(-1, 1)
        y = df['GDP'].values

        model = LinearRegression()
        model.fit(X, y)

        # تحديد عدد السنوات المستقبلية للتنبؤ
        forecast_years = st.slider("📅 Years to Forecast", 1, 10, 5)

        # إنشاء بيانات السنوات المستقبلية
        last_year = df['Year'].max()
        future_years = np.arange(last_year + 1, last_year + forecast_years + 1).reshape(-1, 1)
        predictions = model.predict(future_years)

        forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'GDP': predictions})

        st.subheader("🔮 Forecasted GDP")
        st.dataframe(forecast_df)

        # دمج البيانات الأصلية والمتوقعة للرسم
        combined_df = pd.concat([df, forecast_df], ignore_index=True)

        fig_forecast = px.line(combined_df, x='Year', y='GDP', title='GDP Forecast')
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.error("❌ CSV file must contain 'Year' and 'GDP' columns.")
else:
    st.info("📄 Please upload a CSV file to begin.")
