import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime
import pytz
import os
import io
from timezonefinder import TimezoneFinder

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
# 1. Auto-detect Local Timezone from Coordinates
# -------------------------------------------------
def get_local_timezone(lat, lon):
    """Get timezone name from latitude/longitude"""
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    if tz_name is None:
        tz_name = tf.closest_timezone_at(lat=lat, lng=lon)
    if tz_name is None:
        tz_name = "UTC"  # fallback
    return tz_name

# -------------------------------------------------
# 2. Load Models
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
# 3. Solar Geometry (UTC-based)
# -------------------------------------------------
SOLAR_CONSTANT = 1367

def deg_to_rad(deg):
    return deg * (np.pi / 180)

def calculate_declination(day_of_year):
    return 23.45 * np.sin(deg_to_rad(360 * (284 + day_of_year) / 365))

def calculate_hour_angle(solar_time):
    return 15 * (solar_time - 12)

def solar_time_correction(longitude, standard_meridian, day_of_year):
    B = deg_to_rad(360 * (day_of_year - 81) / 364)
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    return (4 * (longitude - standard_meridian) + eot) / 60

def calculate_extraterrestrial_radiation(latitude, longitude, timestamp_utc):
    day_of_year = timestamp_utc.timetuple().tm_yday
    hour, minute, second = timestamp_utc.hour, timestamp_utc.minute, timestamp_utc.second
    latitude_rad = deg_to_rad(latitude)
    declination = calculate_declination(day_of_year)
    declination_rad = deg_to_rad(declination)
    standard_meridian = 0  # UTC
    solar_time = (hour + minute/60 + second/3600 +
                  solar_time_correction(longitude, standard_meridian, day_of_year))
    hour_angle = calculate_hour_angle(solar_time)
    hour_angle_rad = deg_to_rad(hour_angle)
    cos_zenith_angle = (np.sin(latitude_rad) * np.sin(declination_rad) +
                        np.cos(latitude_rad) * np.cos(declination_rad) *
                        np.cos(hour_angle_rad))
    return SOLAR_CONSTANT * max(cos_zenith_angle, 0)

# -------------------------------------------------
# 4. 24-Hour Prediction (Starts at 6 AM Local)
# -------------------------------------------------
def predict_high_res_next_24_hours(latitude, longitude, start_utc):
    global models, model_names
    end_utc = start_utc + datetime.timedelta(hours=24)
    hours = pd.date_range(start=start_utc, end=end_utc, freq='H', tz='UTC')
    weather_data = pd.DataFrame({
        "datetime": hours,
        "hour_cloudcover": np.random.uniform(0, 100, len(hours)),
        "hour": hours.hour,
        "month": hours.month
    })
    prediction_results = []
    for idx, row in weather_data.iterrows():
        dt = row["datetime"]
        cloudcover = row["hour_cloudcover"]
        hour = row["hour"]
        month = row["month"]
        try:
            extraterrestrial = calculate_extraterrestrial_radiation(latitude, longitude, dt)
            features_df = pd.DataFrame([{
                "extraterrestrial": extraterrestrial,
                "cloudcover": cloudcover,
                "hour": hour,
                "month": month
            }])
            model_predictions = []
            for model in models:
                pred = 0 if (hour <= 5 or hour >= 18) else model.predict(features_df)[0]
                pred = max(pred, 0.0)
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
            st.error(f"Error at {dt}: {e}")
    if not prediction_results:
        st.error("No predictions made.")
        return pd.DataFrame()
    return pd.DataFrame(prediction_results)

# -------------------------------------------------
# 5. Statistics
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
# 6. Energy & 24-Hour Plot
# -------------------------------------------------
def calculate_energy_and_plot(df, date_str):
    interval = 1.0
    df['lower_energy'] = df['Lower_Bound'] * interval
    df['mean_energy'] = df['Mean'] * interval
    df['upper_energy'] = df['Upper_Bound'] * interval
    total_lower = df['lower_energy'].sum()
    total_mean = df['mean_energy'].sum()
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
    # Plot full 24 hours
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
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Irradiance (W/m²)')
    ax.set_title(f'24-Hour Solar Forecast - {date_str}')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return energy_df, fig

# -------------------------------------------------
# 7. Main App - LOCAL 6 AM to 6 PM (Clean)
# -------------------------------------------------
def main():
    st.set_page_config(page_title="Solar Forecast", layout="wide")
    st.title("Solar Irradiance Forecast")
    st.write("Enter coordinates to get a 6 AM to 6 PM local time forecast.")

    col1, col2 = st.columns(2)
    with col1:
        latitude = st.text_input("Latitude", value="35.6762")
    with col2:
        longitude = st.text_input("Longitude", value="139.6503")

    if st.button("Generate Forecast"):
        try:
            lat = float(latitude)
            lon = float(longitude)
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                st.error("Invalid coordinates.")
                return
        except ValueError:
            st.error("Enter numbers only.")
            return

        with st.spinner("Loading models..."):
            load_models()
        if not models:
            st.error("No models found in '34_models_all' folder.")
            return

        # Auto-detect local timezone
        local_tz_name = get_local_timezone(lat, lon)
        local_tz = pytz.timezone(local_tz_name)
        st.info(f"Detected timezone: {local_tz_name}")

        # Start at 6 AM local tomorrow
        tomorrow_local = datetime.datetime.now(local_tz).date() + datetime.timedelta(days=1)
        forecast_start_local = datetime.datetime(
            tomorrow_local.year, tomorrow_local.month, tomorrow_local.day,
            6, 0, tzinfo=local_tz
        )
        forecast_start_utc = forecast_start_local.astimezone(pytz.UTC)
        date_str = forecast_start_local.strftime("%Y-%m-%d")

        with st.spinner("Generating forecast..."):
            forecast_df = predict_high_res_next_24_hours(lat, lon, forecast_start_utc)
            if forecast_df.empty:
                st.error("Failed to generate forecast.")
                return
            forecast_df = add_statistics_columns(forecast_df)
            energy_df, fig = calculate_energy_and_plot(forecast_df, date_str)

        st.pyplot(fig)

        # CSV in local time
        forecast_rows = energy_df[energy_df['datetime'] != 'Total'].copy()
        total_row = energy_df[energy_df['datetime'] == 'Total'].copy()

        forecast_rows['datetime'] = pd.to_datetime(forecast_rows['datetime'], utc=True)
        forecast_rows['datetime'] = forecast_rows['datetime'].dt.tz_convert(local_tz)
        forecast_rows['datetime'] = forecast_rows['datetime'].dt.strftime("%Y-%m-%d %H:%M")

        total_row['datetime'] = 'Total'

        csv_df = pd.concat([forecast_rows, total_row], ignore_index=True)
        csv_df = csv_df[['datetime', 'Lower_Bound', 'Mean', 'Upper_Bound',
                         'lower_energy', 'mean_energy', 'upper_energy']]

        csv_buffer = io.StringIO()
        csv_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"solar_forecast_{date_str}_{local_tz_name}.csv",
            mime="text/csv"
        )

        st.success("Forecast generated successfully.")

if __name__ == "__main__":
    main()
