
# Dashboard Dropout Mahasiswa

Aplikasi dashboard interaktif berbasis *machine learning* untuk menganalisis dan memprediksi risiko dropout mahasiswa. Dibangun menggunakan Streamlit dengan dukungan visualisasi dan analitik prediktif.

---

# Business Understanding

## Permasalahan Bisnis
Jaya Jaya Institut mengalami tingkat dropout mahasiswa yang tinggi dan membutuhkan sistem untuk mendeteksi risiko dropout sejak dini. Tingginya angka dropout memberikan dampak negatif terhadap reputasi institusi serta efektivitas sistem pendidikan. Diperlukan sistem prediksi untuk mengidentifikasi mahasiswa berisiko agar intervensi dapat dilakukan lebih awal dan lebih tepat sasaran.

## Cakupan Proyek
- Pengembangan model *machine learning* untuk memprediksi kemungkinan dropout berdasarkan data siswa.
- Pembuatan dashboard visual untuk memantau performa akademik dan faktor risiko dropout.
- Penyusunan rekomendasi kebijakan berbasis data.

---

# Persiapan

## Sumber Data
Data internal siswa Jaya Jaya Institut, termasuk:
- Profil demografis
- Performa akademik semester awal
- Kondisi sosial ekonomi
- Faktor eksternal lainnya

Dataset mencakup lebih dari 30 fitur dan ribuan catatan mahasiswa.

## Setup Environment

### Prasyarat
- Python ≥ 3.8

### Instalasi Library
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
```

### Tools Visualisasi
- Tableau Public atau Google Looker Studio (opsional untuk eksplorasi lanjutan)

---

# Cara Menjalankan Aplikasi

## Menggunakan Streamlit Cloud
Kunjungi [link aplikasi] (akan diperbarui setelah deployment).

## Menjalankan Secara Lokal
1. Clone repository ini:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run dashboard.py
   ```

## Menggunakan Docker
1. Build image:
   ```bash
   docker build -t dropout-dashboard .
   ```
2. Jalankan container:
   ```bash
   docker run -p 8501:8501 dropout-dashboard
   ```
3. Akses aplikasi di `http://localhost:8501`

---

# Business Dashboard

**Link:** [Contoh Link Tableau / Looker Studio Dashboard]

Dashboard ini memvisualisasikan indikator kunci:
- Distribusi dropout berdasarkan jurusan
- Korelasi antara keterlambatan pembayaran dan dropout
- Performa akademik semester awal dan kaitannya terhadap dropout
- Segmentasi siswa berdasarkan tingkat risiko

---

# Teknologi yang Digunakan

- Python
- Streamlit
- Pandas, Numpy
- Matplotlib, Seaborn, Plotly
- Scikit-learn, XGBoost
- Joblib
- Tableau Public / Google Looker Studio (opsional)

---

# Model Prediksi

Model utama yang digunakan adalah:
- **Support Vector Machine (SVM)**
- Validasi model dilakukan menggunakan metrik akurasi, precision, recall, dan F1-score.

---

# Conclusion

- Model prediksi menunjukkan performa memuaskan (akurasi dan recall tinggi).
- Fitur yang paling berpengaruh terhadap dropout:
  - Keterlambatan pembayaran (*tuition not up-to-date*)
  - Jumlah mata kuliah yang disetujui pada semester awal
  - Latar belakang pendidikan orang tua (khususnya ibu)

---

# Rekomendasi Action Items

- **Intervensi Finansial:** Menyediakan bantuan keuangan atau skema pembayaran fleksibel bagi siswa berisiko.
- **Dukungan Akademik:** Menawarkan mentoring atau bimbingan belajar untuk mahasiswa dengan performa rendah.
- **Pendekatan Sosial:** Memberikan dukungan psikososial kepada mahasiswa dengan latar belakang keluarga berisiko.
- **Monitoring Sistematis:** Mengintegrasikan model ke dalam sistem akademik untuk deteksi dini dan pemantauan berkelanjutan.
