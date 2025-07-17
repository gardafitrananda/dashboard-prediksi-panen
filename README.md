# dashboard-prediksi-panen
Dashboard ini dibuat sebagai Final Project untuk mata kuliah Big Data & Predictive Analytics.

## Deskripsi

Dashboard ini bertujuan untuk memprediksi hasil panen berdasarkan beberapa faktor penting seperti penggunaan pestisida, curah hujan, dan suhu rata-rata. Dengan menggunakan model Regresi Linier, kami menganalisis bagaimana setiap faktor ini berkontribusi terhadap hasil panen.

## Fitur
- **Prediksi Interaktif**: Pengguna dapat mengubah nilai input (suhu, pestisida, dll.) melalui slider untuk melihat hasil prediksi secara langsung.
- **Visualisasi Data**: Menyajikan 6 jenis visualisasi data yang berbeda untuk analisis eksploratif (EDA), termasuk korelasi, tren, dan distribusi data.
- **Evaluasi Model**: Menampilkan metrik evaluasi performa model seperti R-squared, MAE, dan RMSE.
- **Optimasi Prediksi**: Terdapat tombol "Nilai Terbaik" dan "Reset" untuk membantu pengguna menemukan skenario hasil panen tertinggi atau kembali ke nilai awal.

## Cara Menjalankan Proyek Secara Lokal
1.  Pastikan Anda memiliki Python terinstall.
2.  Clone repositori ini.
3.  Install semua library yang dibutuhkan dengan perintah:
    ```
    pip install -r requirements.txt
    ```
4.  Jalankan aplikasi Streamlit dengan perintah:
    ```
    streamlit run dashboard.py
    ```
