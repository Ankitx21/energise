import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import io

# ===== Directory Configuration =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "34_models_all")
V16_COMBINED_DIR = os.path.join(BASE_DIR, "v16_combined")

# Ensure directories exist
for directory in [MODELS_DIR, V16_COMBINED_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== Global Variables =====
models = []
model_names = []

def load_models():
    global models, model_names
    models = []
    model_names = []
    if os.path.exists(MODELS_DIR) and any(f.endswith(".pkl") for f in os.listdir(MODELS_DIR)):
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
        st.error(f"No models found in {MODELS_DIR}. Please add models.")

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

def calculate_extraterrestrial_radiation(latitude, longitude, timezone, timestamp):
    day_of_year = timestamp.timetuple().tm_yday
    hour, minute, second = timestamp.hour, timestamp.minute, timestamp.second
    latitude_rad = deg_to_rad(latitude)
    declination = calculate_declination(day_of_year)
    declination_rad = deg_to_rad(declination)
    standard_meridian = timezone * 15
    solar_time = hour + (minute / 60) + (second / 3600) + solar_time_correction(longitude, standard_meridian, day_of_year)
    hour_angle = calculate_hour_angle(solar_time)
    hour_angle_rad = deg_to_rad(hour_angle)
    cos_zenith_angle = (
        np.sin(latitude_rad) * np.sin(declination_rad) +
        np.cos(latitude_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )
    return SOLAR_CONSTANT * max(cos_zenith_angle, 0)

def predict_high_res_next_24_hours(latitude, longitude, start_time):
    global models, model_names
    timezone = 5.5  # Assuming IST; adjust if needed
    end_time = start_time + timedelta(hours=24)
    hours = pd.date_range(start=start_time, end=end_time, freq='H')
    # Simulate weather data (replace with API call if available)
    weather_data = pd.DataFrame({
        "datetime": hours,
        "hour_cloudcover": np.random.uniform(0, 100, len(hours)),  # Placeholder
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
            extraterrestrial = calculate_extraterrestrial_radiation(
                latitude, longitude, timezone, dt
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
        st.error("No predictions made. Check models or input data.")
        return pd.DataFrame()
    forecast_df = pd.DataFrame(prediction_results)
    forecast_df["datetime"] = forecast_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return forecast_df

def add_statistics_columns(df):
    model_columns = [col for col in df.columns if col.lower().startswith('model_') and col.lower().endswith('_pred')]
    if not model_columns:
        df['Mean'] = float('nan')
        df['Standard_Deviation'] = float('nan')
        df['Lower_Bound'] = float('nan')
        df['Upper_Bound'] = float('nan')
    else:
        df[model_columns] = df[model_columns].apply(pd.to_numeric, errors='coerce')
        df['Mean'] = df[model_columns].mean(axis=1)
        df['Standard_Deviation'] = df[model_columns].std(axis=1)
        df['Lower_Bound'] = df['Mean'] - 1.96 * df['Standard_Deviation']
        df['Upper_Bound'] = df['Mean'] + 1.96 * df['Standard_Deviation']
        df['Mean'] = df['Mean'].clip(lower=0)
        df['Lower_Bound'] = df['Lower_Bound'].clip(lower=0)
        df['Upper_Bound'] = df['Upper_Bound'].clip(lower=0)
    return df

def calculate_energy_and_plot(df, date_str):
    time_interval = 1.0  # 1 hour
    df['lower_energy'] = df['Lower_Bound'] * time_interval
    df['mean_energy'] = df['Mean'] * time_interval
    df['upper_energy'] = df['Upper_Bound'] * time_interval
    total_lower = df['lower_energy'].sum()
    total_mean = df['mean_energy'].sum()
    total_upper = df['upper_energy'].sum()
    energy_df = df[['datetime', 'Lower_Bound', 'Mean', 'Upper_Bound', 'lower_energy', 'mean_energy', 'upper_energy']].copy()
    totals_row = pd.DataFrame({
        'datetime': ['Total'],
        'Lower_Bound': [np.nan],
        'Mean': [np.nan],
        'Upper_Bound': [np.nan],
        'lower_energy': [total_lower],
        'mean_energy': [total_mean],
        'upper_energy': [total_upper]
    })
    energy_df = pd.concat([energy_df, totals_row], ignore_index=True)
    # Plotting
    plot_df = energy_df[:-1]  # Exclude totals row
    plot_df['datetime'] = pd.to_datetime(plot_df['datetime'])
    plot_df = plot_df[(plot_df['datetime'].dt.hour >= 5) & (plot_df['datetime'].dt.hour <= 19)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df['datetime'], plot_df['Lower_Bound'], label=f'Lower Bound (Energy: {total_lower:.2f} unit-hours)', color='#1f77b4')
    ax.plot(plot_df['datetime'], plot_df['Mean'], label=f'Mean (Energy: {total_mean:.2f} unit-hours)', color='#2ca02c')
    ax.plot(plot_df['datetime'], plot_df['Upper_Bound'], label=f'Upper Bound (Energy: {total_upper:.2f} unit-hours)', color='#d62728')
    ax.fill_between(plot_df['datetime'], plot_df['Lower_Bound'], plot_df['Upper_Bound'], color='#ff7f0e', alpha=0.3, label='Prediction Band')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Irradiance')
    ax.set_title('Irradiance Forecast with Prediction Bands')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return energy_df, fig

def main():
    st.title("Solar Irradiance Forecast")
    st.write("Enter your location coordinates to get a 24-hour solar irradiance forecast.")
    
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.text_input("Latitude", value="12.92142594422952")
    with col2:
        longitude = st.text_input("Longitude", value="77.43551506633048")
    
    if st.button("Generate Forecast"):
        try:
            latitude = float(latitude)
            longitude = float(longitude)
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                st.error("Invalid coordinates. Latitude must be [-90, 90], Longitude [-180, 180].")
                return
        except ValueError:
            st.error("Please enter valid numeric coordinates.")
            return
        
        load_models()
        if not models:
            st.error("No models loaded. Please ensure models are in the '34_models_all' directory.")
            return
        
        forecast_time = datetime.now().date() + timedelta(days=1)
        forecast_time = datetime(forecast_time.year, forecast_time.month, forecast_time.day, 0, 0)
        date_str = forecast_time.strftime("%Y-%m-%d")
        
        with st.spinner("Generating forecast..."):
            forecast_df = predict_high_res_next_24_hours(latitude, longitude, forecast_time)
            if forecast_df.empty:
                st.error("Failed to generate forecast.")
                return
            
            forecast_df = add_statistics_columns(forecast_df)
            energy_df, fig = calculate_energy_and_plot(forecast_df, date_str)
            
            st.pyplot(fig)
            
            csv_buffer = io.StringIO()
            energy_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv_buffer.getvalue(),
                file_name=f"energy_values_{date_str}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()