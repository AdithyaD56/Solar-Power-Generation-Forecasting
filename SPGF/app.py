import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from prophet import Prophet
from datetime import datetime

st.set_page_config(page_title="Solar Power Dashboard", layout="wide")

# ------------------- DEFAULT FILE PATHS -------------------
DEFAULT_PATH = Path(r"C:\Users\Adithya DVDM\Downloads\SPGF")
DEFAULT_GEN = DEFAULT_PATH / "Plant_2_Generation_Data.csv"
DEFAULT_WEATHER = DEFAULT_PATH / "Plant_2_Weather_Sensor_Data.csv"

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data(gen_file, weather_file):
    gen = pd.read_csv(gen_file)
    weather = pd.read_csv(weather_file)

    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'])
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'])

    df = pd.merge(gen, weather, on=["DATE_TIME", "PLANT_ID"], how="inner")

    if df.empty:
        st.error("⚠ Data merge returned no results. Ensure both CSV files belong to Plant 2 data.")
        st.stop()

    df['DATE'] = df['DATE_TIME'].dt.date
    df['YEAR'] = df['DATE_TIME'].dt.year
    df['MONTH'] = df['DATE_TIME'].dt.month
    df['WEEK'] = df['DATE_TIME'].dt.isocalendar().week
    return df

# Sidebar Upload Section
st.sidebar.title("⚙️ Data Options")
gen_file = st.sidebar.file_uploader("Upload Generation Data CSV", type=["csv"])
weather_file = st.sidebar.file_uploader("Upload Weather Data CSV", type=["csv"])

if gen_file is None: gen_file = DEFAULT_GEN
if weather_file is None: weather_file = DEFAULT_WEATHER

df = load_data(gen_file, weather_file)

# ------------------- HEADER -------------------
st.title("☀️ Solar Power Generation Dashboard")
st.caption(f"📅 System Date: **{datetime.now().date()}**")

# ------------------- NAVIGATION MENU -------------------
page = st.sidebar.radio("Navigate", ["Overview", "Weather Impact", "Forecast Output", "Future Energy Forecast"])


# ------------------- OVERVIEW PAGE -------------------
if page == "Overview":
    st.subheader("🔆 Power Generation Overview")

    st.write("### 📅 Select Time Range")
    option = st.selectbox("View data for:", ["Today", "This Week", "This Month", "This Year", "Custom Range"])

    latest_date = df['DATE'].max()

    if option == "Today":
        filtered = df[df['DATE'] == latest_date]
        display_range = f"{latest_date}"
    elif option == "This Week":
        week = df['WEEK'].max()
        filtered = df[df['WEEK'] == week]
        display_range = f"Week {int(week)}"
    elif option == "This Month":
        month = df['MONTH'].max()
        year = df['YEAR'].max()
        filtered = df[(df['MONTH'] == month) & (df['YEAR'] == year)]
        display_range = f"{month}/{year}"
    elif option == "This Year":
        year = df['YEAR'].max()
        filtered = df[df['YEAR'] == year]
        display_range = f"{year}"
    else:
        start = st.date_input("Start Date", df['DATE'].min())
        end = st.date_input("End Date", df['DATE'].max())
        filtered = df[(df['DATE'] >= start) & (df['DATE'] <= end)]
        display_range = f"{start} → {end}"

    if filtered.empty:
        st.warning("⚠ No data available for the selected range.")
        st.stop()

    total_energy = filtered['DAILY_YIELD'].sum()
    avg_power = filtered['AC_POWER'].mean()

    col1, col2 = st.columns(2)
    col1.metric("Total Energy (kWh)", f"{total_energy:.2f}")
    col2.metric("Average AC Power (kW)", f"{avg_power:.2f}")

    st.caption(f"📌 Displaying data for: **{display_range}**")

    st.markdown("---")
    fig = px.line(filtered.sort_values("DATE_TIME"), x="DATE_TIME", y="AC_POWER", color_discrete_sequence=["#FF8C00"])
    st.plotly_chart(fig, use_container_width=True)


# ------------------- WEATHER IMPACT PAGE -------------------
elif page == "Weather Impact":
    st.subheader("🌦 Weather Influence on Power Output")

    df["HOUR"] = df["DATE_TIME"].dt.hour
    hourly = df.groupby("HOUR")[["AC_POWER", "IRRADIATION", "MODULE_TEMPERATURE"]].mean().reset_index()

    fig = px.line(hourly, x="HOUR", y=["AC_POWER", "IRRADIATION", "MODULE_TEMPERATURE"], title="Hourly Trend")
    st.plotly_chart(fig, use_container_width=True)

    st.info("✅ More sunlight increases output\n⚠ Overheating reduces panel efficiency")


# ------------------- INSTANT POWER FORECAST PAGE -------------------
elif page == "Forecast Output":
    st.subheader("🔮 Instant Power Prediction")

    irr = st.slider("Irradiation (W/m²)", 0, 1200, 600)
    module = st.slider("Module Temperature (°C)", 20, 85, 45)

    prediction = max((irr * 0.85) - (module * 0.15), 0)

    if st.button("Predict Power"):
        st.success(f"Estimated AC Power Output: **{prediction:.2f} kW**")


# ------------------- FUTURE ENERGY FORECAST PAGE (PLAYFUL MODE) -------------------
elif page == "Future Energy Forecast":
    st.subheader("📅 Future Energy Forecast")

    daily = df.groupby("DATE")[["DAILY_YIELD"]].sum().reset_index()
    prophet_df = daily.rename(columns={"DATE": "ds", "DAILY_YIELD": "y"})

    @st.cache_resource
    def train_prophet():
        model = Prophet()
        model.fit(prophet_df)
        return model

    model = train_prophet()
    future = model.make_future_dataframe(periods=365 * 5)
    forecast = model.predict(future)
    forecast["DATE"] = forecast["ds"].dt.date

    mode = st.radio("Select Forecast Mode:", ["Single Date", "Date Range"])

    # Conversions (fixed)
    home_rate = 2.5
    ev_rate = 7.5
    class_rate = 1.2

    # -------- SINGLE DATE --------
    if mode == "Single Date":
        selected = st.date_input("Select a date:")
        if st.button("Predict Power"):
            row = forecast[forecast["DATE"] == selected]
            if not row.empty:
                energy = row["yhat"].values[0]

                homes = energy / home_rate
                evs = energy / ev_rate
                classes = energy / class_rate

                st.success(f"🔮 Expected Energy on {selected}: **{energy:.2f} kWh**")
                st.write(f"""
                **This energy can be used to power approximately:**
                - 🏠 **{homes:.1f} homes**
                - 🚗 **{evs:.1f} electric vehicle fast chargers**
                - 🎒 **{classes:.1f} classrooms**
                """)

    # -------- DATE RANGE --------
    else:
        colA, colB = st.columns(2)
        with colA: start = st.date_input("Start Date")
        with colB: end = st.date_input("End Date")

        if st.button("Predict Power for Range"):
            selected_range = forecast[(forecast["DATE"] >= start) & (forecast["DATE"] <= end)]
            if not selected_range.empty:
                total = selected_range["yhat"].sum()

                homes = total / home_rate
                evs = total / ev_rate
                classes = total / class_rate

                st.success(f"🔮 Total Expected Energy from {start} to {end}: **{total:.2f} kWh**")
                st.write(f"""
                **This amount of energy can roughly support:**
                - 🏠 **{homes:.1f} homes**
                - 🚗 **{evs:.1f} EV fast charging stations**
                - 🎒 **{classes:.1f} classrooms**
                """)

                fig = px.line(selected_range, x="DATE", y="yhat", title="📈 Forecast Trend")
                st.plotly_chart(fig, use_container_width=True)
