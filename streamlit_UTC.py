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

# ----------------------------------------------------------------------
# Directory configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "34_models_all")
for directory in [MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ----------------------------------------------------------------------
# Global variables
# ----------------------------------------------------------------------
models = []
model_names = []

# ----------------------------------------------------------------------
# 1. Auto-detect local timezone
# ----------------------------------------------------------------------
def get_local_timezone(lat, lon):
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    if tz_name is None:
        tz_name = tf.closest_timezone_at(lat=lat, lng=lon)
    return tz_name or "UTC"

# ----------------------------------------------------------------------
# 2. Load all .pkl models
# ----------------------------------------------------------------------
def load_models():
    global models, model_names
    models, model_names = [], []
    if not os.path.isdir(MODELS_DIR):
        st.error(f"Folder {MODELS_DIR} not found.")
        return
    for fn in os.listdir(MODELS_DIR):
        if fn.lower().endswith(".pkl"):
            path = os.path.join(MODELS_DIR, fn)
            try:
                m = joblib.load(path)
                models.append(m)
                model_names.append(fn.replace(".pkl", ""))
            except Exception as e:
                st.error(f"Error loading {path}: {e}")
    if models:
        st.success(f"Loaded {len(models)} models.")
    else:
        st.error("No .pkl files found in 34_models_all.")

# ----------------------------------------------------------------------
# 3. Solar geometry (UTC based)
# ----------------------------------------------------------------------
SOLAR_CONSTANT = 1367

def deg_to_rad(deg):
    return deg * np.pi / 180

def declination(doy):
    return 23.45 * np.sin(deg_to_rad(360 * (284 + doy) / 365))

def solar_time_correction(lon, doy):
    B = deg_to_rad(360 * (doy - 81) / 364)
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    return (4 * lon + eot) / 60

def extraterrestrial_and_day_flag(lat, lon, utc_dt):
    """Return (extraterrestrial, is_day) for a UTC datetime."""
    doy = utc_dt.timetuple().tm_yday
    h, m, s = utc_dt.hour, utc_dt.minute, utc_dt.second
    lat_r = deg_to_rad(lat)

    solar_t = h + m/60 + s/3600 + solar_time_correction(lon, doy)
    ha = deg_to_rad(15 * (solar_t - 12))

    dec = declination(doy)
    dec_r = deg_to_rad(dec)

    cos_z = (np.sin(lat_r) * np.sin(dec_r) +
             np.cos(lat_r) * np.cos(dec_r) * np.cos(ha))
    ext = SOLAR_CONSTANT * max(cos_z, 0)
    is_day = cos_z > 0
    return ext, is_day

# ----------------------------------------------------------------------
# 4. 24-hour prediction – starts at 00:00 local time
# ----------------------------------------------------------------------
def predict_24h(lat, lon, start_utc):
    global models, model_names
    end_utc = start_utc + datetime.timedelta(hours=24)
    hours = pd.date_range(start=start_utc, end=end_utc, freq='H', tz='UTC')

    rows = []
    for utc_dt in hours:
        ext, is_day = extraterrestrial_and_day_flag(lat, lon, utc_dt)
        cloud = np.random.uniform(0, 100)

        feats = pd.DataFrame([{
            "extraterrestrial": ext,
            "cloudcover": cloud,
            "hour": utc_dt.hour,
            "month": utc_dt.month
        }])

        preds = []
        for m in models:
            p = m.predict(feats)[0] if is_day else 0.0
            preds.append(max(p, 0.0))

        mean = np.mean(preds)
        rows.append({
            "datetime": utc_dt,
            "extraterrestrial": round(ext, 2),
            "cloudcover": round(cloud, 2),
            "Mean": round(mean, 2),
            "Lower_Bound": round(mean - 1.96 * np.std(preds), 2),
            "Upper_Bound": round(mean + 1.96 * np.std(preds), 2),
            "lower_energy": round(mean - 1.96 * np.std(preds), 2),
            "mean_energy": round(mean, 2),
            "upper_energy": round(mean + 1.96 * np.std(preds), 2),
        })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# 5. Plot in local time
# ----------------------------------------------------------------------
def plot_local(df, date_str, tz_name):
    df_plot = df.copy()
    df_plot['datetime'] = pd.to_datetime(df_plot['datetime'], utc=True)
    df_plot['datetime'] = df_plot['datetime'].dt.tz_convert(tz_name)

    total_lower = df_plot['lower_energy'].sum()
    total_mean  = df_plot['mean_energy'].sum()
    total_upper = df_plot['upper_energy'].sum()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_plot['datetime'], df_plot['Lower_Bound'],
            label=f'Lower Bound (Energy: {total_lower:.1f} Wh/m²)', color='#1f77b4')
    ax.plot(df_plot['datetime'], df_plot['Mean'],
            label=f'Mean (Energy: {total_mean:.1f} Wh/m²)', color='#2ca02c')
    ax.plot(df_plot['datetime'], df_plot['Upper_Bound'],
            label=f'Upper Bound (Energy: {total_upper:.1f} Wh/m²)', color='#d62728')
    ax.fill_between(df_plot['datetime'],
                    df_plot['Lower_Bound'], df_plot['Upper_Bound'],
                    color='#ff7f0e', alpha=0.3, label='95% Prediction Band')
    ax.set_xlabel(f'Time ({tz_name})')
    ax.set_ylabel('Irradiance (W/m²)')
    ax.set_title(f'24-Hour Solar Forecast – {date_str}')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig, total_lower, total_mean, total_upper

# ----------------------------------------------------------------------
# 6. UI
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Solar Forecast", layout="wide")
    st.title("Solar Irradiance Forecast")
    st.write("Enter coordinates → 24-hour forecast in **local time** (00:00–23:00).")

    col1, col2 = st.columns(2)
    with col1:
        lat_in = st.text_input("Latitude", value="35.6762")
    with col2:
        lon_in = st.text_input("Longitude", value="139.6503")

    if st.button("Generate Forecast"):
        try:
            lat = float(lat_in)
            lon = float(lon_in)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                st.error("Invalid coordinates.")
                return
        except ValueError:
            st.error("Enter numeric values.")
            return

        load_models()
        if not models:
            return

        # ---- local timezone ------------------------------------------------
        tz_name = get_local_timezone(lat, lon)
        tz = pytz.timezone(tz_name)
        st.info(f"Detected timezone: **{tz_name}**")

        # ---- start at 00:00 local tomorrow --------------------------------
        tomorrow = datetime.datetime.now(tz).date() + datetime.timedelta(days=1)
        start_local = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day,
                                       0, 0, tzinfo=tz)
        start_utc = start_local.astimezone(pytz.UTC)
        date_str = start_local.strftime("%Y-%m-%d")

        with st.spinner("Predicting 24 h..."):
            df = predict_24h(lat, lon, start_utc)

        # ---- plot -----------------------------------------------------------
        fig, tot_l, tot_m, tot_u = plot_local(df, date_str, tz_name)
        st.pyplot(fig)

        # ---- CSV in local time ---------------------------------------------
        df_csv = df.copy()
        df_csv['datetime'] = pd.to_datetime(df_csv['datetime'], utc=True)
        df_csv['datetime'] = df_csv['datetime'].dt.tz_convert(tz_name)
        df_csv['datetime'] = df_csv['datetime'].dt.strftime("%Y-%m-%d %H:%M")

        # add total row
        total_row = pd.DataFrame([{
            "datetime": "Total",
            "Lower_Bound": np.nan, "Mean": np.nan, "Upper_Bound": np.nan,
            "lower_energy": tot_l, "mean_energy": tot_m, "upper_energy": tot_u
        }])
        df_csv = pd.concat([df_csv, total_row], ignore_index=True)
        df_csv = df_csv[['datetime', 'Lower_Bound', 'Mean', 'Upper_Bound',
                         'lower_energy', 'mean_energy', 'upper_energy']]

        csv_io = io.StringIO()
        df_csv.to_csv(csv_io, index=False)
        st.download_button(
            label="Download CSV (local time)",
            data=csv_io.getvalue(),
            file_name=f"solar_forecast_{date_str}_{tz_name}.csv",
            mime="text/csv"
        )

        st.success("Forecast ready – 24 h from midnight local time.")

if __name__ == "__main__":
    main()
