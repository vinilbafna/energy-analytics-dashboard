import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Energy Analytics Dashboard", layout="wide")
st.title("⚡ Household Energy Consumption Analytics")
st.markdown("**1.4M+ records | Random Forest Model | R² = 0.9989**")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    df = pd.read_csv(url, sep=";", na_values=["?"], low_memory=False,
                     parse_dates={"DateTime": ["Date", "Time"]},
                     dayfirst=True)
    df.dropna(inplace=True)
    df.set_index("DateTime", inplace=True)
    # Normalize all column names
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df["Hour"] = df.index.hour
    df["Month"] = df.index.month
    df["DayOfWeek"] = df.index.dayofweek
    return df

@st.cache_resource
def train_model(df):
    target = [c for c in df.columns if 'active_power' in c.lower()][0]
    features = ["Global_reactive_power", "Voltage", "Global_intensity",
                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
                "Hour", "Month", "DayOfWeek"]
    # Only use features that exist
    features = [f for f in features if f in df.columns]
    sample = df.sample(100000, random_state=42)
    X = sample[features]
    y = sample[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, r2_score(y_test, preds), mean_absolute_error(y_test, preds), target, features

with st.spinner("⏳ Loading 1.4M records from UCI..."):
    df = load_data()

# Show actual column names for debugging
st.write("Columns found:", df.columns.tolist())

with st.spinner("🤖 Training Random Forest model..."):
    model, r2, mae, target, features = train_model(df)

# ── KPI Cards ──────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Active Power", f"{df[target].mean():.3f} kW")
col2.metric("Avg Voltage", f"{df['Voltage'].mean():.2f} V")
col3.metric("Model R²", f"{r2:.4f}")
col4.metric("Model MAE", f"{mae:.4f} kW")

st.markdown("---")

# ── Daily Trend ────────────────────────────────────────────────────
st.subheader("📈 Daily Average Power Consumption")
daily = df[target].resample("D").mean()
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(daily, color="steelblue", linewidth=0.8)
ax.set_ylabel("Power (kW)")
st.pyplot(fig)

# ── Hourly Pattern ─────────────────────────────────────────────────
st.subheader("🕐 Peak Usage Hours")
hourly = df.groupby("Hour")[target].mean()
fig2, ax2 = plt.subplots(figsize=(10, 3))
sns.barplot(x=hourly.index, y=hourly.values, palette="Blues_d", ax=ax2)
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Avg Power (kW)")
st.pyplot(fig2)

# ── Heatmap ────────────────────────────────────────────────────────
st.subheader("🔥 Feature Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# ── ML Predictor ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 Predict Energy Consumption")

col1, col2, col3 = st.columns(3)
voltage    = col1.slider("Voltage (V)", 220.0, 255.0, 240.0)
intensity  = col2.slider("Global Intensity (A)", 0.2, 48.0, 5.0)
reactive   = col3.slider("Reactive Power (kW)", 0.0, 1.4, 0.1)

col4, col5, col6 = st.columns(3)
hour  = col4.selectbox("Hour of Day", list(range(24)), index=12)
month = col5.selectbox("Month", list(range(1, 13)), index=6)
dow   = col6.selectbox("Day of Week (0=Mon)", list(range(7)), index=0)

sub1 = st.slider("Sub Metering 1 (Kitchen)", 0.0, 88.0, 1.0)
sub2 = st.slider("Sub Metering 2 (Laundry)", 0.0, 80.0, 1.0)
sub3 = st.slider("Sub Metering 3 (AC/Heater)", 0.0, 31.0, 17.0)

if st.button("⚡ Predict Power"):
    input_data = pd.DataFrame(
        [[reactive, voltage, intensity, sub1, sub2, sub3, hour, month, dow]],
        columns=features)
    prediction = model.predict(input_data)[0]
    st.success(f"⚡ Predicted Global Active Power: **{prediction:.4f} kW**")

st.markdown("---")
st.caption("Built by Vinil Bafna | EI, Nirma University | Dataset: UCI Household Power Consumption")
