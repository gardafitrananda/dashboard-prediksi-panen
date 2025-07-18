import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Prediksi Panen Padi",
    page_icon="🌾",
    layout="wide"
)

# --- FUNGSI UNTUK MEMUAT DATA & MELATIH MODEL ---
@st.cache_data
def load_and_train():
    try:
        # --- PERUBAHAN: Membaca kembali dataset LAMA ---
        df = pd.read_csv('rice_paddy_crop_yield.csv')
    except FileNotFoundError:
        st.error("File 'rice_paddy_crop_yield.csv' tidak ditemukan.")
        st.stop()

    df = df.loc[:, ~df.columns.duplicated()]
    df_renamed = df.rename(columns={
        'Yield_Value': 'yield', 'Pesticide_Value': 'pesticides',
        'Avg_Rainfall_mm': 'rainfall', 'Avg_Temperature_celsius': 'temperature'
    })
    
    # --- PERUBAHAN: Menyesuaikan kolom yang dihapus untuk dataset LAMA ---
    cols_to_drop = ['Country', 'Crop_Item']
    df_processed = df_renamed.drop(columns=cols_to_drop, errors='ignore')
    
    cols_to_process = [col for col in df_processed.columns if pd.api.types.is_numeric_dtype(df_processed[col])]
    df_numeric = df_processed[cols_to_process].copy()
    df_clean = df_numeric.dropna()

    if 'yield' not in df_clean.columns:
        st.error("Kolom 'yield' tidak ditemukan.")
        st.stop()

    # --- Model 1: Menggunakan SEMUA fitur ---
    X_full = df_clean.drop(columns=['yield', 'Year'], errors='ignore')
    y_full = df_clean['yield']
    model_full = LinearRegression().fit(X_full, y_full)
    
    # Proses evaluasi untuk model SEMUA fitur
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    model_eval = LinearRegression().fit(X_train, y_train)
    y_pred_eval = model_eval.predict(X_test)
    r2 = metrics.r2_score(y_test, y_pred_eval)
    mae = metrics.mean_absolute_error(y_test, y_pred_eval)
    mse = metrics.mean_squared_error(y_test, y_pred_eval)
    rmse = np.sqrt(mse)
    eval_metrics = {"R-squared (R²)": r2, "MAE (ton/ha)": mae, "MSE": mse, "RMSE (ton/ha)": rmse}

    # Data untuk plot "Prediksi vs Aktual (Semua Fitur)"
    y_pred_full_plot = model_full.predict(X_full)
    df_results_full = pd.DataFrame({'Aktual': y_full, 'Prediksi': y_pred_full_plot})
    
    # --- Model 2: Hanya menggunakan 3 FITUR (seperti di Colab) ---
    features_3 = ['pesticides', 'rainfall', 'temperature']
    X_3_features = df_clean[features_3]
    y_3_features = df_clean['yield']
    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3_features, y_3_features, test_size=0.2, random_state=42)
    model_3_features = LinearRegression().fit(X_train_3, y_train_3)
    
    # Data untuk plot "Prediksi vs Aktual (Model 3 Fitur)"
    y_pred_3_features_plot = model_3_features.predict(X_test_3)
    df_results_3_features = pd.DataFrame({'Aktual': y_test_3, 'Prediksi': y_pred_3_features_plot})
    
    return model_full, X_full.columns, df_clean, df, df_results_full, eval_metrics, df_results_3_features

model, feature_names, df_clean, df_original, df_results_full, eval_metrics, df_results_3_features = load_and_train()

# --- INISIALISASI SESSION STATE ---
if 'init' not in st.session_state:
    for feature in feature_names:
        st.session_state[feature] = float(df_clean[feature].mean())
    st.session_state['init'] = True

# --- FUNGSI CALLBACK ---
def set_best_values():
    for i, feature in enumerate(feature_names):
        if model.coef_[i] > 0: st.session_state[feature] = float(df_clean[feature].max())
        else: st.session_state[feature] = float(df_clean[feature].min())
def reset_values():
    for feature in feature_names: st.session_state[feature] = float(df_clean[feature].mean())

# --- SIDEBAR ---
st.sidebar.header("⚙️ Panel Kontrol Prediksi")
for feature in feature_names:
    min_val, max_val = float(df_clean[feature].min()), float(df_clean[feature].max())
    label_text = feature.replace('_', ' ').title()
    if 'rainfall' in feature: label_text += ' (mm/tahun)'
    elif 'pesticides' in feature: label_text += ' (ton)'
    elif 'temperature' in feature: label_text += ' (°C)'
    st.sidebar.slider(label=label_text, min_value=min_val, max_value=max_val, key=feature)
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
col1.button("Nilai Terbaik 📈", on_click=set_best_values, use_container_width=True, help="Atur slider untuk prediksi hasil panen tertinggi.")
col2.button("Reset 🔄", on_click=reset_values, use_container_width=True, help="Kembalikan semua slider ke nilai rata-rata.")

# --- HALAMAN UTAMA ---
st.title("🌾 Dashboard Prediksi Hasil Panen Padi")
st.header("💡 Hasil Prediksi Interaktif")
current_inputs = {feature: st.session_state[feature] for feature in feature_names}
input_df = pd.DataFrame([current_inputs])
prediction = model.predict(input_df)[0]
# --- PERUBAHAN: Menyesuaikan satuan ke ton/ha ---
st.metric(label="Prediksi dari Input Anda", value=f"{prediction:,.2f} ton/ha")
st.caption("Satuan hasil panen (yield) pada dataset ini adalah ton/ha.")

st.markdown("---")
st.header("📈 Hasil Evaluasi Model (Model dengan Semua Fitur)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R-squared (R²)", f"{eval_metrics['R-squared (R²)']:.4f}")
col2.metric("MAE (ton/ha)", f"{eval_metrics['MAE (ton/ha)']:,.2f}")
col3.metric("MSE", f"{eval_metrics['MSE']:,.2f}")
col4.metric("RMSE (ton/ha)", f"{eval_metrics['RMSE (ton/ha)']:,.2f}")

st.markdown("---")
st.header("📊 Visualisasi Data & Model")

plot_choice = st.selectbox(
    "Pilih grafik untuk ditampilkan:",
    ("Prediksi vs. Aktual (Semua Fitur)", "Prediksi vs. Aktual (Model 3 Fitur)", "Korelasi Antar Variabel", "Tren Hasil Panen per Tahun", "Hasil Panen vs Pestisida", "Distribusi Suhu", "Distribusi Curah Hujan")
)

# --- VISUALISASI ---
if plot_choice == "Prediksi vs. Aktual (Semua Fitur)":
    fig = px.scatter(
        df_results_full, x='Aktual', y='Prediksi', opacity=0.7,
        title='Prediksi vs. Aktual (Model dengan Semua Fitur)',
        labels={'Aktual': 'Hasil Aktual (ton/ha)', 'Prediksi': 'Hasil Prediksi (ton/ha)'}
    )
    min_val, max_val = min(df_results_full.min().min(), df_results_full.min().min()), max(df_results_full.max().max(), df_results_full.max().max())
    fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color='Red', width=2, dash='dash'))
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Prediksi vs. Aktual (Model 3 Fitur)":
    fig = px.scatter(
        df_results_3_features, x='Aktual', y='Prediksi', opacity=0.7,
        title='Prediksi vs. Aktual (Model Regresi Linier Berganda - 3 Fitur)',
        labels={'Aktual': 'Hasil Aktual (ton/ha)', 'Prediksi': 'Hasil Prediksi (ton/ha)'}
    )
    min_val, max_val = min(df_results_3_features.min().min(), df_results_3_features.min().min()), max(df_results_3_features.max().max(), df_results_3_features.max().max())
    fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color='Red', width=2, dash='dash'))
    st.plotly_chart(fig, use_container_width=True)
    
elif plot_choice == "Korelasi Antar Variabel":
    fig = px.imshow(df_original.corr(numeric_only=True), text_auto=".2f", color_continuous_scale='RdBu_r', title="Korelasi Antar Variabel")
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Tren Hasil Panen per Tahun":
    df_trend = df_original.groupby('Year')['Yield_Value'].mean().reset_index()
    fig = px.line(df_trend, x="Year", y="Yield_Value", markers=True, title="Tren Rata-rata Hasil Panen per Tahun", labels={"Yield_Value": "Rata-rata Hasil Panen (ton/ha)"})
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Hasil Panen vs Pestisida":
    fig = px.scatter(df_original, x="Pesticide_Value", y="Yield_Value", color="Year", title="Hasil Panen vs Penggunaan Pestisida", labels={"Pesticide_Value": "Penggunaan Pestisida (ton)", "Yield_Value": "Hasil Panen (ton/ha)"}, color_continuous_scale='Viridis', trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Distribusi Suhu":
    fig = px.histogram(df_original, x="Avg_Temperature_celsius", nbins=15, title="Distribusi Suhu Rata-rata (°C)")
    st.plotly_chart(fig, use_container_width=True)
    
elif plot_choice == "Distribusi Curah Hujan":
    fig = px.histogram(df_original, x="Avg_Rainfall_mm", nbins=15, title="Distribusi Curah Hujan Rata-rata (mm/tahun)")
    st.plotly_chart(fig, use_container_width=True)