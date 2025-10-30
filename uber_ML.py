from datetime import date
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import xgboost as xgb
from sklearn.preprocessing import RobustScaler as robust
from sklearn.preprocessing import PolynomialFeatures
import numpy.polynomial.polynomial as poly
import datetime

with open('uber_price_pred_model.pkl', 'rb') as fr:
    model = pickle.load(fr)
with open('scaler_sc_X.pkl', 'rb') as fr:
    scaler = pickle.load(fr)


# Weekend checker
def is_weekend(dt):
    return 1 if dt.weekday() >= 5 else 0  # 5 = Sabtu, 6 = Minggu

# One Hot Time Category
def encode_hour_category(hour):
    return [
        1 if 6 <= hour < 9 else 0,
        1 if 9 <= hour < 16 else 0,
        1 if 16 <= hour < 19 else 0,
        1 if 19 <= hour <= 23 else 0,
        1 if hour < 6 else 0
    ]

def run_uber_ML():
    passenger_count = st.number_input("Jumlah Penumpang", 1, 6, key='1')
    distance_km = st.number_input("Jarak perjalanan (KM)", 1, 999, key='2')
    tanggal = st.date_input('Tanggal Pick up', value = date.today())
    waktu = st.time_input('Waktu Pick up')
    
    dt = datetime.datetime.combine(tanggal, waktu)
    hour = dt.hour

    weekend = is_weekend(dt)
    hour_ohe = encode_hour_category(hour)

    # 12 Fitur sesuai training
    features = np.array([[
        passenger_count,
        distance_km,
        hour,
        dt.day,
        dt.month,
        dt.year,
        weekend,
        *hour_ohe
    ]])

    
    if st.button("Prediksi tarif"):
        try:
            features_scaled = scaler.transform(features)

            pred = model.predict(features_scaled)[0]
            st.success(f"Tarif diperkirakan: **${pred:.2f} USD**")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Pastikan urutan fitur sudah sesuai model.")
        
if __name__ =='__main__':
    run_uber_ML()
