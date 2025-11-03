import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import io
from zoneinfo import ZoneInfo   # Python 3.9+  (fallback to pytz if needed)

# ===== Directory Configuration =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "34_models_all")
V16_COMBINED_DIR = os.path.join(BASE_DIR, "v16_combined")
for directory in [MODELS_DIR, V16_COMBINED_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== Global Variables =====
models = []
model_names = []

# -------------------------------------------------
# 1. Model loading (unchanged)
# -------------------------------------------------
def load_models():
    global models, model_names
    models = []
    model_names = []
    if os.path.isdir(MODELS_DIR) and any(f.endswith(".pkl") for f in os.listdir(MODELS_DIR)):
        for file_name in os.listdir(MODELS_DIR):
            if file_name.endswith(".pkl"):
                model_path = os.path.join(MODELS_DIR, file_name)
                try:
                    model = joblib.load(model_path)
                    models.append(model)
                    model_name = file_name.replace(".pkl", "")
                    model_names.append(model_name)
                except Exception as e:
                    st.error(f"Error loading model {model_path}: {e}")
        st.success(f"Loaded {len(models)} models from {MODELS_DIR}")
    else:
        st.error(f"No models found in {MODELS_DIR}. Please add .pkl files.")

# -------------------------------------------------
# 2. Solar geometry (unchanged)
# -------------------------------------------------
SOLAR_CONSTANT = 1367
def deg_to_rad(deg): return deg * (np.pi / 180)

def calculate_declination(day_of_year):
    return 23.45 * np.sin(deg_to_rad(360 * (284 + day_of_year) / 365))

def calculate_hour_angle(solar_time):
    return 15 * (solar_time - 12)

def solar_time_correction(longitude, standard_meridian, day_of_year):
    B = deg_to_rad(360 * (day_of_year - 81) / 364)
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    return (4 * (longitude - standard_meridian) + eot) / 60

def calculate_extraterrestrial_radiation(latitude, longitude, timezone_offset, timestamp):
    """
    timezone_offset : hours from UTC (e.g. 0 for UTC, 5.5 for IST)
    """
    day_of_year = timestamp.timetuple().tm_yday
    hour, minute, second = timestamp.hour, timestamp.minute, timestamp.second
    latitude_rad = deg_to_rad(latitude)
    declination = calculate_declination(day_of_year)
    declination_rad = deg_to_rad(declination)

    standard_meridian = timezone_offset * 15          # degrees per hour
    solar_time = (hour + minute/60 + second/3600 +
                  solar_time_correction(longitude, standard_meridian, day_of_year))
    hour_angle = calculate_hour_angle(solar_time)
    hour_angle_rad = deg_to_rad(hour_angle)

    cos_zenith_angle = (np.sin(latitude_rad) * np.sin(declination_rad) +
                        np.cos(latitude_rad) * np.cos(declination_rad) *
                        np.cos(hour_angle_rad))
    return SOLAR_CONSTANT * max(cos_zenith_angle, 0)

# -------------------------------------------------
# 3. 24-h prediction – now UTC based
# -------------------------------------------------
def predict_high_res_next_24_hours(latitude, longitude, start_utc):
    global models, model_names

    # ----- UTC is used everywhere -----
    timezone_offset = 0.0                     # 0 h = UTC
    end_utc = start_utc + timedelta(hours=24)
    hours = pd.date_range(start=start_utc, end=end_utc, freq='H', tz='UTC')

    # placeholder weather (replace with real API later)
    weather_data = pd.DataFrame({
        "datetime": hours,
        "hour_cloudcover": np.random.uniform(0, 100, len(hours)),
        "hour": hours.hour,
        "month": hours.month
    })

    prediction_results = []
    for idx, row in weather_data.iterrows():
        dt = row["datetime"]                     # aware UTC datetime
        cloudcover = row["hour_cloudcover"]
        hour = row["hour"]
        month = row["month"]

        try:
            extraterrestrial = calculate_extraterrestrial_radiation(
                latitude, longitude, timezone_offset, dt
            )
            features_df = pd.DataFrame([{
                "extraterrestrial": extraterrestrial,
                "cloudcover": cloudcover,
                "hour": hour,
                "month": month
            }])

            model_predictions = []
            for model in models:
                pred = 0 if (hour <= 5 or hour >= 18) else model.predict(features_df)[0]
                pred = max(pred, 0.0)          # clip only negatives
                model_predictions.append(pred)

            mean_pred = np.mean(model_predictions)
            max_pred = np.max(model_predictions)
            min_pred = np.min(model_predictions)
            max_model_idx = np.argmax(model_predictions)
            min_model_idx = np.argmin(model_predictions)

            result = {
                "datetime": dt,
                "cloudcover": round(cloudcover, 2),
                "extraterrestrial": round(extraterrestrial, 2),
                "mean_predicted_irradiance": round(mean_pred, 2),
                "max_predicted_irradiance": round(max_pred, 2),
                "min_predicted_irradiance": round(min_pred, 2),
                "max_model": model_names[max_model_idx],
                "min_model": model_names[min_model_idx],
                "hour": hour,
                "month": month
            }
            for i, pred in enumerate(model_predictions):
                result[f"model_{i+1}_pred"] = round(pred, 2)

            prediction_results.append(result)
        except Exception as e:
            st.error(f"Error at index {idx}, time {dt}: {e}")

    if not prediction_results:
        st.error("No predictions made.")
        return pd.DataFrame()

    forecast_df = pd.DataFrame(prediction_results)
    # Keep aware UTC timestamps for the CSV; display will be converted later
    forecast_df["datetime"] = forecast_df["datetime"].dt.tz_convert(None)
    forecast_df["datetime"] = forecast_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return forecast_df

# -------------------------------------------------
# 4. Statistics – clip only negatives
# -------------------------------------------------
def add_statistics_columns(df):
    model_cols = [c for c in df.columns if c.lower().startswith('model_') and c.lower().endswith('_pred')]
    if not model_cols:
        df['Mean'] = np.nan
        df['Standard_Deviation'] = np.nan
        df['Lower_Bound'] = np.nan
        df['Upper_Bound'] = np.nan
    else:
        df[model_cols] = df[model_cols].apply(pd.to_numeric, errors='coerce')
        df['Mean'] = df[model_cols].mean(axis=1)
        df['Standard_Deviation'] = df[model_cols].std(axis=1)
        df['Lower_Bound'] = df['Mean'] - 1.96 * df['Standard_Deviation']
        df['Upper_Bound'] = df['Mean'] + 1.96 * df['Standard_Deviation']
        df['Mean'] = df['Mean'].clip(lower=0)
        df['Lower_Bound'] = df['Lower_Bound'].clip(lower=0)
        df['Upper_Bound'] = df['Upper_Bound'].clip(lower=0)
    return df

# -------------------------------------------------
# 5. Energy & 24-h plot (full day)
# -------------------------------------------------
def calculate_energy_and_plot(df, date_str):
    interval = 1.0
    df['lower_energy'] = df['Lower_Bound'] * interval
    df['mean_energy']  = df['Mean']       * interval
    df['upper_energy'] = df['Upper_Bound'] * interval

    total_lower = df['lower_energy'].sum()
    total_mean  = df['mean_energy'].sum()
    total_upper = df['upper_energy'].sum()

    energy_df = df[['datetime', 'Lower_Bound', 'Mean', 'Upper_Bound',
                    'lower_energy', 'mean_energy', 'upper_energy']].copy()

    totals_row = pd.DataFrame({
        'datetime': ['Total'],
        'Lower_Bound': [np.nan], 'Mean': [np.nan], 'Upper_Bound': [np.nan],
        'lower_energy': [total_lower],
        'mean_energy': [total_mean],
        'upper_energy': [total_upper]
    })
    energy_df = pd.concat([energy_df, totals_row], ignore_index=True)

    # ----- Plot full 24 h -----
    plot_df = energy_df[:-1].copy()
    plot_df['datetime'] = pd.to_datetime(plot_df['datetime'])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df['datetime'], plot_df['Lower_Bound'],
            label=f'Lower Bound (Energy: {total_lower:.2f} Wh/m²)', color='#1f77b4')
    ax.plot(plot_df['datetime'], plot_df['Mean'],
            label=f'Mean (Energy: {total_mean:.2f} Wh/m²)', color='#2ca02c')
    ax.plot(plot_df['datetime'], plot_df['Upper_Bound'],
            label=f'Upper Bound (Energy: {total_upper:.2f} Wh/m²)', color='#d62728')
    ax.fill_between(plot_df['datetime'],
                    plot_df['Lower_Bound'], plot_df['Upper_Bound'],
                    color='#ff7f0e', alpha=0.3, label='95% Prediction Band')

    ax.set_xlabel('Datetime (UTC)')
    ax.set_ylabel('Irradiance (W/m²)')
    ax.set_title('24-Hour Solar Irradiance Forecast (UTC) with Prediction Bands')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return energy_df, fig

# -------------------------------------------------
# 6. Streamlit UI – optional local-time display
# -------------------------------------------------
def main():
    st.title("Solar Irradiance Forecast – UTC Based")
    st.write("Enter coordinates to get a **full-day (UTC)** irradiance forecast.")

    col1, col2 = st.columns(2)
    with col1:
        latitude = st.text_input("Latitude", value="12.92142594422952")
    with col2:
        longitude = st.text_input("Longitude", value="77.43551506633048")

    # Optional: let user pick a display time-zone
    tz_options = ["UTC"] + [tz for tz in ZoneInfo.available_timezones() if "Asia" in tz or "Europe" in tz or "America" in tz]
    display_tz = st.selectbox("Display times in:", options=tz_options, index=0)

    if st.button("Generate Forecast"):
        try:
            lat = float(latitude)
            lon = float(longitude)
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                st.error("Invalid coordinates.")
                return
        except ValueError:
            st.error("Enter numeric coordinates.")
            return

        load_models()
        if not models:
            st.error("No models in '34_models_all'.")
            return

        # Forecast starts at 00:00 UTC of the *next* calendar day
        tomorrow_utc = datetime.now(tz=ZoneInfo("UTC")).date() + timedelta(days=1)
        forecast_start_utc = datetime(tomorrow_utc.year, tomorrow_utc.month,
                                      tomorrow_utc.day, 0, 0, tzinfo=ZoneInfo("UTC"))
        date_str = forecast_start_utc.strftime("%Y-%m-%d")

        with st.spinner("Generating 24 h UTC forecast..."):
            forecast_df = predict_high_res_next_24_hours(lat, lon, forecast_start_utc)
            if forecast_df.empty:
                st.error("Failed to generate forecast.")
                return

            forecast_df = add_statistics_columns(forecast_df)
            energy_df, fig = calculate_energy_and_plot(forecast_df, date_str)

            # ---- Show plot (times are in UTC) ----
            st.pyplot(fig)

            # ---- Optional: convert CSV & display to chosen TZ ----
            csv_df = energy_df.copy()
            csv_df['datetime'] = pd.to_datetime(csv_df['datetime'])
            if display_tz != "UTC":
                csv_df['datetime'] = csv_df['datetime'].dt.tz_localize("UTC")\
                                                       .dt.tz_convert(display_tz)\
                                                       .dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                csv_df['datetime'] = csv_df['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")

            csv_buffer = io.StringIO()
            csv_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv_buffer.getvalue(),
                file_name=f"energy_values_{date_str}_UTC.csv",
                mime="text/csv"
            )

            # Show a small note
            st.info(f"Forecast is calculated in **UTC**. "
                    f"CSV times are shown in **{display_tz}** if you chose a different zone.")

if __name__ == "__main__":
    main()