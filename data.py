import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from pmdarima import auto_arima

# ====== إعداد الصفحة ======
st.set_page_config(page_title="تحليل الناتج المحلي", layout="wide")

# الوضع الليلي
dark_mode = st.sidebar.checkbox("🌙 تفعيل الوضع الليلي")
if dark_mode:
    st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

# اللغة والاسم
lang = "ar" if st.sidebar.radio("🌐 اختر اللغة", ["العربية", "English"]) == "العربية" else "en"
user_name = st.sidebar.text_input("👤 أدخل اسمك:", "زائر" if lang == "ar" else "Visitor")
st.success(f"مرحبًا بك يا {user_name}" if lang == "ar" else f"Welcome, {user_name}!")

# ====== تحميل البيانات ======
@st.cache_data
def load_data():
    years = list(range(2000, 2023))
    data = {"Country Name": ["UAE", "Egypt", "KSA"]}
    for year in years:
        data[str(year)] = [100000 + (year - 2000) * 15000 + (i * 5000) for i in range(3)]
    return pd.DataFrame(data)

df = load_data()
df_long = df.melt(id_vars=["Country Name"], var_name="Year", value_name="GDP")
df_long["Year"] = df_long["Year"].astype(int)

# ====== إدخال يدوي ======
st.markdown("### ➕ إدخال يدوي للبيانات")
manual_country = st.selectbox("اختر الدولة", ["UAE", "Egypt", "KSA"], key="manual_country")
manual_year = st.number_input("السنة", min_value=1990, max_value=2100, value=2025, key="manual_year")
manual_gdp = st.number_input("الناتج المحلي (GDP)", min_value=0, value=500000, step=10000, key="manual_gdp")

if st.button("إضافة البيانات يدويًا"):
    new_row = {"Country Name": manual_country, "Year": int(manual_year), "GDP": manual_gdp}
    df_long = pd.concat([df_long, pd.DataFrame([new_row])], ignore_index=True)
    st.success("✅ تم إضافة البيانات بنجاح!")

# ====== اختيار الدول ======
countries = st.multiselect("🌍 اختر الدول:", df_long["Country Name"].unique(), default=["UAE", "Egypt", "KSA"])
filtered_df = df_long[df_long["Country Name"].isin(countries)]

# ====== الرسوم ======
st.plotly_chart(px.line(filtered_df, x="Year", y="GDP", color="Country Name", markers=True, title="GDP Over Time"))

# ====== توقعات متقدمة مع إرسال بريد إلكتروني ======
from sklearn.tree import DecisionTreeRegressor
from pmdarima.arima import auto_arima

selected_country = st.selectbox("🔮 اختر دولة للتوقع:", countries)
country_df = filtered_df[filtered_df["Country Name"] == selected_country].sort_values("Year")
X = country_df["Year"].values.reshape(-1, 1)
y = country_df["GDP"].values

# Linear Regression
lr_model = LinearRegression().fit(X, y)
future_years = np.arange(2023, 2031).reshape(-1, 1)
lr_preds = lr_model.predict(future_years)

# Decision Tree
tree_model = DecisionTreeRegressor().fit(X, y)
tree_preds = tree_model.predict(future_years)

# ARIMA
arima_model = auto_arima(y, seasonal=False, suppress_warnings=True)
arima_preds = arima_model.predict(n_periods=8)

# DataFrame للتوقعات
forecast_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Linear Regression": lr_preds.astype(int),
    "Decision Tree": tree_preds.astype(int),
    "ARIMA": arima_preds.astype(int)
})
####################################################################################################
#####################################################################################################
# ====== توقعات لدولة واحدة ======
from sklearn.tree import DecisionTreeRegressor
from pmdarima import auto_arima
import io
from fpdf import FPDF

selected_country = st.selectbox("اختر دولة للتوقع:" if lang == "ar" else "Select country to forecast:", countries)
country_df = filtered_df[filtered_df["Country Name"] == selected_country].sort_values("Year")
X = country_df["Year"].values.reshape(-1, 1)
y = country_df["GDP"].values

# ====== Linear Regression ======
lr_model = LinearRegression().fit(X, y)
future_years = np.arange(2023, 2031).reshape(-1, 1)
lr_preds = lr_model.predict(future_years)

# ====== Decision Tree ======
tree_model = DecisionTreeRegressor().fit(X, y)
tree_preds = tree_model.predict(future_years)

# ====== ARIMA ======
arima_model = auto_arima(y, seasonal=False, suppress_warnings=True)
arima_preds = arima_model.predict(n_periods=8)

# ====== تجميع التوقعات ======
forecast_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Linear Regression": lr_preds.astype(int),
    "Decision Tree": tree_preds.astype(int),
    "ARIMA": arima_preds.astype(int)
})

st.markdown(f"### 🔮 توقع الناتج المحلي لـ {selected_country}")
st.plotly_chart(
    px.line(forecast_df, x="Year", y=["Linear Regression", "Decision Tree", "ARIMA"],
            title=f"GDP Forecast for {selected_country}"),
    use_container_width=True
)

st.dataframe(forecast_df.style.format({"Linear Regression": "{:,}", "Decision Tree": "{:,}", "ARIMA": "{:,}"}))

# ====== إنشاء ملفات التقرير من التوقعات ======
def generate_forecast_reports(forecast_df, country):
    # توليد ملف Excel
    excel_bytes = io.BytesIO()
    with pd.ExcelWriter(excel_bytes, engine="xlsxwriter") as writer:
        forecast_df.to_excel(writer, index=False, sheet_name="Forecast")
    excel_bytes.seek(0)

    # توليد ملف PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"GDP Forecast Report - {country}", ln=True, align="C")
    pdf.ln(10)

    for idx, row in forecast_df.iterrows():
        pdf.cell(200, 10, txt=f"Year {int(row['Year'])}: "
                              f"LR = {int(row['Linear Regression']):,}, "
                              f"DT = {int(row['Decision Tree']):,}, "
                              f"ARIMA = {int(row['ARIMA']):,}", ln=True)

    # تحويل الإخراج إلى BytesIO
    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_bytes = io.BytesIO(pdf_output)

    return excel_bytes, pdf_bytes


# ====== أزرار تحميل التقرير ======
excel_bytes, pdf_bytes = generate_forecast_reports(forecast_df, selected_country)

st.download_button("⬇️ تحميل التوقعات بصيغة Excel", excel_bytes, file_name="GDP_Forecast.xlsx")
st.download_button("⬇️ تحميل التوقعات بصيغة PDF", pdf_bytes, file_name="GDP_Forecast.pdf")

####################################################################################################
#####################################################################################################


# عرض التوقعات
st.markdown(f"### 🔮 توقع الناتج المحلي لـ {selected_country}")
st.plotly_chart(
    px.line(forecast_df, x="Year", y=["Linear Regression", "Decision Tree", "ARIMA"],
            title=f"GDP Forecast for {selected_country}")
)
st.dataframe(forecast_df)

# ====== إرسال التوقعات بالبريد ======
recipient = st.text_input("📧 البريد الإلكتروني للمستلم:")
sender_email = st.text_input("✉️ بريدك الإلكتروني (Gmail فقط):")
sender_password = st.text_input("🔑 كلمة مرور التطبيق (App Password):", type="password")

def send_forecast_email(recipient, sender_email, sender_password, forecast_df, country):
    msg = EmailMessage()
    msg["Subject"] = f"GDP Forecast Report - {country}"
    msg["From"], msg["To"] = sender_email, recipient
    msg.set_content(f"Attached is the GDP forecast report for {country}.")

    # توليد ملف Excel للتوقعات
    excel_data = io.BytesIO()
    with pd.ExcelWriter(excel_data, engine="xlsxwriter") as writer:
        forecast_df.to_excel(writer, index=False)
    msg.add_attachment(excel_data.getvalue(), maintype="application",
                       subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       filename=f"{country}_GDP_Forecast.xlsx")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        st.success("✅ تم إرسال التوقعات بنجاح!")
    except Exception as e:
        st.error(f"❌ فشل الإرسال: {e}")

if st.button("📤 إرسال التوقعات عبر البريد"):
    if recipient and sender_email and sender_password:
        send_forecast_email(recipient, sender_email, sender_password, forecast_df, selected_country)
    else:
        st.warning("⚠️ الرجاء إدخال جميع بيانات البريد.")
