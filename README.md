# 🫁 TBScan - Deteksi Tuberculosis dari Citra X-Ray

Final Project Pengolahan Citra Digital - Universitas Gadjah Mada

## 👥 Anggota Kelompok

| Nama | NIM |
|------|-----|
| Kosmas Rio Legowo | 23/512012/PA/21863 |
| Kukuh Agus Hermawan | 24/533395/PA/22573 |
| Danar Fathurahman | 24/538200/PA/22828 |
| Mikail Achmad | 24/542370/PA/23026 |

---

## 📌 Deskripsi Proyek

**TBScan** adalah aplikasi berbasis web untuk mendeteksi penyakit Tuberculosis (TB) dari gambar rontgen dada (Chest X-Ray). Aplikasi ini menggunakan teknik pengolahan citra digital dan machine learning untuk menganalisis tekstur paru-paru.

### Fitur Utama:
- Upload gambar X-Ray (JPG, PNG, JPEG)
- Segmentasi otomatis area paru-paru
- Ekstraksi fitur tekstur menggunakan GLCM
- Klasifikasi TB vs Normal menggunakan SVM

---

## 🛠️ Teknologi yang Digunakan

- **Python** - Bahasa pemrograman utama
- **OpenCV** - Pengolahan citra
- **Scikit-learn** - Machine learning (SVM, PCA)
- **Scikit-image** - Ekstraksi fitur GLCM
- **Streamlit** - Web app framework
- **NumPy** - Komputasi numerik

---

## 📂 Struktur Folder

```
Final-Project-PCD/
├── Final_Project_PCD.ipynb    # Notebook training model
├── README.md
└── Streamlit/
    ├── app.py                 # Aplikasi web Streamlit
    ├── tb_features.py         # Modul ekstraksi fitur
    └── tb_svm_pipeline.joblib # Model SVM yang sudah dilatih
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/kosmasrio0411/Final-Project-PCD.git
cd Final-Project-PCD
```

### 2. Install Dependencies
```bash
pip install streamlit opencv-python scikit-learn scikit-image numpy joblib
```

### 3. Jalankan Aplikasi
```bash
cd Streamlit
streamlit run app.py
```

### 4. Buka di Browser
Aplikasi akan terbuka di `http://localhost:8501`

---

## 📊 Metodologi

### Pipeline Pengolahan Citra:
1. **Preprocessing** - Resize, CLAHE, Gaussian Blur
2. **Segmentasi** - Otsu Thresholding + Morphological Operations
3. **Ekstraksi Fitur** - GLCM (Contrast, Correlation, Homogeneity, Energy, Entropy, Variance) + Fitur Statistik
4. **Reduksi Dimensi** - PCA (95% variance)
5. **Klasifikasi** - SVM dengan kernel RBF

### Dataset:
- **TBX-11K** dari Kaggle
- Binary classification: TB vs Non-TB

---

## 📸 Screenshot

*Upload gambar X-Ray → Segmentasi Paru → Ekstraksi Fitur → Hasil Prediksi*

---

## ⚠️ Disclaimer

Aplikasi ini dibuat untuk **tujuan akademik**. Hasil prediksi **bukan pengganti diagnosis medis profesional**. Selalu konsultasikan dengan dokter spesialis untuk diagnosis yang akurat.

---

## 📄 Lisensi

© 2025 Kelompok 1 PCD - Universitas Gadjah Mada
