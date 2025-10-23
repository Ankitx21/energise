# removing interpolation of 15 minutes, making only next day 24 hrs predictions

import requests
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

LOCATION = "12.92142594422952,77.43551506633048"  # Latitude,Longitude

api_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/12.92142594422952%2C77.43551506633048?key=DQWTAPH3779BKWQ24RFGQZTSH&contentType=json"


# ===== Global Variables =====
models = []
model_names = []

# ===== Directory Configuration =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "34_models_all")
V16_COMBINED_DIR = os.path.join(BASE_DIR, "v16_combined")

# Ensure directories exist
for directory in [MODELS_DIR, V16_COMBINED_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"Ensured directory exists: {directory}")

def load_models():
    global models, model_names
    models = []
    model_names = []
    
    if os.path.exists(MODELS_DIR) and any(f.endswith(".pkl") for f in os.listdir(MODELS_DIR)):
        print(f"Loading models from {MODELS_DIR}...")
        for file_name in os.listdir(MODELS_DIR):
            if file_name.endswith(".pkl"):
                model_path = os.path.join(MODELS_DIR, file_name)
                try:
                    model = joblib.load(model_path)
                    models.append(model)
                    model_name = file_name.replace(".pkl", "")
                    model_names.append(model_name)
                    print(f"Loaded model: {model_name}")
                except Exception as e:
                    print(f"Error loading model {model_path}: {e}")
    else:
        print(f"No models found in {MODELS_DIR}. Exiting.")
    print(f"Successfully loaded {len(models)} models from 34_models_all")

def add_sunlight_flag(df):
    df["sun_is_up"] = df["hour"].apply(lambda h: 1 if 5 <= h <= 19 else 0)
    return df

def get_weather_data(start_date, end_date):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            print(f"Fetched weather data from {start_date} to {end_date}.")
            all_data = []
            for day in weather_data.get("days", []):
                day_info = {
                    "day_datetime": day.get("datetime"),
                    "day_cloudcover": day.get("cloudcover"),
                    "day_solarradiation": day.get("solarradiation"),
                }
                for hour in day.get("hours", []):
                    hour_info = {
                        "hour_time": hour.get("datetime"),
                        "hour_cloudcover": hour.get("cloudcover"),
                        "hour_solarradiation": hour.get("solarradiation"),
                    }
                    merged_info = {**day_info, **hour_info}
                    all_data.append(merged_info)
            df = pd.DataFrame(all_data)
            def enrich_time_features(row):
                try:
                    timestamp = f"{row['day_datetime']} {row['hour_time']}"
                    parsed_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    row["datetime"] = parsed_time
                    row["hour"] = parsed_time.hour
                    row["month"] = parsed_time.month
                    row["sin_hour"] = np.sin(2 * np.pi * parsed_time.hour / 24)
                    row["cos_hour"] = np.cos(2 * np.pi * parsed_time.hour / 24)
                except Exception as e:
                    print(f"Error enriching row: {e}")
                return row
            df = df.apply(enrich_time_features, axis=1)
            df = add_sunlight_flag(df)
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
            csv_filename = os.path.join(V16_COMBINED_DIR, "weather_data.csv")
            df.to_csv(csv_filename, index=False)
            print(f"CSV saved with enriched features: {csv_filename}")
            return df
        except requests.exceptions.RequestException as e:
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("Max retries reached. Returning None.")
                return None

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

def predict_high_res_next_24_hours(weather_df, start_time):
    global models, model_names
    print(f"Generating hourly forecast for next 24 hours starting from {start_time}")
    end_time = start_time + timedelta(hours=24)
    future_data = weather_df[
        (weather_df["datetime"] >= start_time) &
        (weather_df["datetime"] <= end_time)
    ].copy()
    if future_data.empty:
        print("No weather data found for the specified period.")
        return pd.DataFrame()
    prediction_results = []
    latitude, longitude = map(float, LOCATION.split(','))
    timezone = 5.5  # Assuming IST; adjust if needed
    for idx, row in future_data.iterrows():
        dt = row["datetime"]
        cloudcover = row["hour_cloudcover"]
        hour = dt.hour
        month = dt.month
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
            print(f"Error at index {idx}, time {dt}: {e}")
    if not prediction_results:
        print("No predictions were made. Check your models or input data.")
        return pd.DataFrame()
    high_res_df = pd.DataFrame(prediction_results)
    high_res_df["datetime"] = high_res_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return high_res_df

def add_statistics_columns(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded DataFrame from {input_file} with shape: {df.shape}")
    except Exception as e:
        print(f"Failed to read input file {input_file}: {str(e)}")
        raise

    print(f"Available columns in DataFrame: {df.columns.tolist()}")
    model_columns = [col for col in df.columns if col.lower().startswith('model_') and col.lower().endswith('_pred')]
    print(f"Found {len(model_columns)} model prediction columns: {model_columns}")

    if len(model_columns) == 0:
        print("No model prediction columns found. Statistics will be NaN or empty.")
        df['Mean'] = float('nan')
        df['Standard_Deviation'] = float('nan')
        df['Lower_Bound'] = float('nan')
        df['Upper_Bound'] = float('nan')
    else:
        try:
            df[model_columns] = df[model_columns].apply(pd.to_numeric, errors='coerce')
            if df[model_columns].isnull().any().any():
                print("Non-numeric values found in model columns, converted to NaN.")
                print(f"Rows with NaN after conversion:\n{df[model_columns][df[model_columns].isnull().any(axis=1)]}")
        except Exception as e:
            print(f"Error converting model columns to numeric: {str(e)}")
            raise

        df['Mean'] = df[model_columns].mean(axis=1)
        df['Standard_Deviation'] = df[model_columns].std(axis=1)
        df['Lower_Bound'] = df['Mean'] - 1.96 * df['Standard_Deviation']
        df['Upper_Bound'] = df['Mean'] + 1.96 * df['Standard_Deviation']

        df['Mean'] = df['Mean'].clip(lower=0)
        df['Lower_Bound'] = df['Lower_Bound'].clip(lower=0)
        df['Upper_Bound'] = df['Upper_Bound'].clip(lower=0)

        print(f"Sample Mean values (first 5 rows): {df['Mean'].head().tolist()}")
        print(f"Sample Standard_Deviation values (first 5 rows): {df['Standard_Deviation'].head().tolist()}")
        print(f"Sample Lower_Bound values (first 5 rows): {df['Lower_Bound'].head().tolist()}")
        print(f"Sample Upper_Bound values (first 5 rows): {df['Upper_Bound'].head().tolist()}")

        if df[['Mean', 'Standard_Deviation', 'Lower_Bound', 'Upper_Bound']].isnull().any().any():
            print("Some statistics contain NaN values. Check input data for issues.")
            print(f"Rows with NaN in statistics:\n{df[df[['Mean', 'Standard_Deviation', 'Lower_Bound', 'Upper_Bound']].isnull().any(axis=1)]}")
        if not df[['Mean', 'Standard_Deviation', 'Lower_Bound', 'Upper_Bound']].replace([float('inf'), -float('inf')], float('nan')).notnull().all().all():
            print("Some statistics contain infinite values. Check input data for extreme values.")

    try:
        df.to_csv(output_file, index=False)
        print(f"Statistics columns added and saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save output file {output_file}: {str(e)}")
        raise

def calculate_energy_and_save(output_path, date_str):
    df = pd.read_csv(output_path)
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
    energy_path = os.path.join(V16_COMBINED_DIR, f"energy_values_{date_str}.csv")
    energy_df.to_csv(energy_path, index=False)
    print(f"Energy values and totals saved to: {energy_path}")
    plot_df = energy_df[:-1]  # Exclude totals row for plotting
    plot_df['datetime'] = pd.to_datetime(plot_df['datetime'])
    plot_df = plot_df[(plot_df['datetime'].dt.hour >= 5) & (plot_df['datetime'].dt.hour <= 19)]
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df['datetime'], plot_df['Lower_Bound'], label=f'Lower Bound (Energy: {total_lower:.2f} unit-hours)', color='#1f77b4')
    plt.plot(plot_df['datetime'], plot_df['Mean'], label=f'Mean (Energy: {total_mean:.2f} unit-hours)', color='#2ca02c')
    plt.plot(plot_df['datetime'], plot_df['Upper_Bound'], label=f'Upper Bound (Energy: {total_upper:.2f} unit-hours)', color='#d62728')
    plt.fill_between(plot_df['datetime'], plot_df['Lower_Bound'], plot_df['Upper_Bound'], color='#ff7f0e', alpha=0.3, label=f'Prediction Band (Total Energy: {total_mean:.2f} unit-hours)')
    plt.xlabel('Datetime')
    plt.ylabel('Irradiance')
    plt.title('Irradiance Forecast with Prediction Bands')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(V16_COMBINED_DIR, f"irradiance_forecast_{date_str}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Chart generated and saved to {plot_path}. Open the PNG file to view.")

def main():
    load_models()
    if not models:
        print("No models loaded. Exiting.")
        return
    
    # Set forecast to next day's midnight
    current_date = datetime.now().date()
    forecast_date = current_date + timedelta(days=1)
    forecast_time = datetime(forecast_date.year, forecast_date.month, forecast_date.day, 0, 0)
    
    # Check if date is within 15 days from today
    max_forecast_date = current_date + timedelta(days=15)
    if forecast_date > max_forecast_date:
        print(f"Forecast date {forecast_date} is beyond 15-day forecast range. Exiting.")
        return
    
    start_date = forecast_time.strftime('%Y-%m-%d')
    end_date = (forecast_time + timedelta(days=2)).strftime('%Y-%m-%d')  # Buffer for 24 hours
    
    print(f"Generating forecast starting from {forecast_time}")
    weather_data = get_weather_data(start_date, end_date)
    if weather_data is None:
        print("Failed to fetch weather data. Exiting.")
        return
    
    forecast_df = predict_high_res_next_24_hours(weather_data, forecast_time)
    date_str = forecast_time.strftime("%Y-%m-%d")
    official_path = os.path.join(V16_COMBINED_DIR, f"next_24_hrs_v16_68_{date_str}.csv")
    forecast_df.to_csv(official_path, index=False)
    print(f"Forecast saved to: {official_path}")
    
    output_path = os.path.join(V16_COMBINED_DIR, f"sd_mean_lb_24_hrs_v16_68_{date_str}.csv")
    add_statistics_columns(official_path, output_path)
    
    calculate_energy_and_save(output_path, date_str)

if __name__ == "__main__":
    main()