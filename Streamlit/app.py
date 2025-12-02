import streamlit as st
import numpy as np
import cv2
import joblib
import os

# --- IMPORT MODUL FITUR ---
# Pastikan file tb_features.py ada di folder yang sama
try:
    import tb_features as fp
except ImportError:
    st.error("⚠️ Modul 'tb_features.py' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="TBScan - Deteksi Tuberculosis",
    page_icon="🫁",
    layout="centered"
)

# ==============================================================================
# FUNGSI LOAD MODEL
# ==============================================================================
@st.cache_resource
def load_pipeline():
    """Load model SVM pipeline dari file joblib."""
    model_path = os.path.join(os.path.dirname(__file__), "tb_svm_pipeline.joblib")
    try:
        pipe = joblib.load(model_path)
        return pipe["scaler"], pipe["pca"], pipe["svm"]
    except FileNotFoundError:
        return None, None, None

# ==============================================================================
# HEADER APLIKASI
# ==============================================================================
st.title("🫁 TBScan - Deteksi Tuberculosis")
st.caption("Sistem Deteksi Tuberculosis Berbasis Citra X-Ray menggunakan SVM + GLCM")

st.divider()

# ==============================================================================
# LOAD MODEL
# ==============================================================================
scaler, pca, svm_clf = load_pipeline()

if scaler is None:
    st.error("❌ Model `tb_svm_pipeline.joblib` tidak ditemukan!")
    st.info("Pastikan file model berada di folder yang sama dengan `app.py`.")
    st.stop()

st.success("✅ Model berhasil dimuat!")

# ==============================================================================
# STEP 1: UPLOAD GAMBAR
# ==============================================================================
st.header("📤 Step 1: Upload Citra X-Ray")
st.write("Upload gambar rontgen dada (Chest X-Ray) dalam format JPG, JPEG, atau PNG.")

uploaded = st.file_uploader(
    "Pilih file gambar X-Ray",
    type=["jpg", "jpeg", "png"],
    help="Upload citra rontgen dada untuk dianalisis"
)

# Variabel untuk menyimpan hasil segmentasi
seg = None
img_gray = None

if uploaded is not None:
    # Baca file gambar
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    st.divider()
    
    # ==============================================================================
    # STEP 2: TAMPILKAN GAMBAR ASLI & SEGMENTASI
    # ==============================================================================
    st.header("🖼️ Step 2: Preview Citra")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Citra Asli")
        st.image(img_gray, caption=f"Resolusi: {img_gray.shape[1]} x {img_gray.shape[0]} px", use_container_width=True)
    
    # Proses Segmentasi
    with col2:
        st.subheader("Segmentasi Paru")
        try:
            with st.spinner("Memproses segmentasi paru..."):
                pre = fp.preprocess_image(img_gray)
                seg, mask = fp.segment_lungs_final(pre)
            
            st.image(seg, caption="Area paru yang tersegmentasi", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal melakukan segmentasi: {e}")
            seg = None
    
    st.divider()
    
    # ==============================================================================
    # STEP 3: EKSTRAKSI FITUR
    # ==============================================================================
    st.header("📊 Step 3: Ekstraksi Fitur GLCM")
    
    if seg is not None:
        with st.expander("ℹ️ Tentang Fitur GLCM", expanded=False):
            st.write("""
            **GLCM (Gray Level Co-occurrence Matrix)** adalah metode ekstraksi fitur tekstur yang menganalisis 
            hubungan spasial antar piksel dalam citra. Fitur yang diekstraksi meliputi:
            
            - **Contrast**: Mengukur variasi intensitas lokal
            - **Correlation**: Mengukur ketergantungan linear antar piksel
            - **Homogeneity**: Mengukur kedekatan distribusi elemen GLCM ke diagonal
            - **Energy**: Mengukur keseragaman tekstur
            - **Entropy**: Mengukur kompleksitas/keacakan tekstur
            - **Variance**: Mengukur dispersi nilai intensitas
            
            Selain itu, fitur statistik intensitas juga diekstraksi:
            - Mean, Std, RMS, Variance, Smoothness, Kurtosis, Skewness
            """)
        
        # Ekstraksi fitur
        feats = fp.glcm_stat_features(seg)
        
        st.write(f"**Jumlah fitur yang diekstraksi:** {len(feats)} fitur")
        
        with st.expander("🔢 Lihat Nilai Fitur", expanded=False):
            st.write(feats)
    
    st.divider()
    
    # ==============================================================================
    # STEP 4: PREDIKSI
    # ==============================================================================
    st.header("🔬 Step 4: Prediksi dengan SVM")
    
    if seg is not None:
        if st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True):
            
            with st.spinner("Menganalisis citra..."):
                # Reshape fitur untuk prediksi
                feats_reshaped = feats.reshape(1, -1)
                
                # Standarisasi fitur
                X_std = scaler.transform(feats_reshaped)
                
                # Reduksi dimensi dengan PCA
                X_pca = pca.transform(X_std)
                
                # Prediksi dengan SVM
                pred = svm_clf.predict(X_pca)[0]
                
                # Hitung jarak ke hyperplane (confidence score)
                dist_score = 0.0
                if hasattr(svm_clf, "decision_function"):
                    dist_score = svm_clf.decision_function(X_pca)[0]
            
            st.divider()
            
            # ==============================================================================
            # STEP 5: HASIL DIAGNOSIS
            # ==============================================================================
            st.header("📋 Step 5: Hasil Diagnosis")
            
            is_tb = (pred == 1)
            
            if is_tb:
                st.error("⚠️ **TERINDIKASI TUBERCULOSIS**")
                st.write("""
                Sistem mendeteksi pola tekstur abnormal yang diasosiasikan dengan Tuberculosis.
                
                **Rekomendasi:**
                - Segera konsultasikan dengan dokter spesialis paru
                - Lakukan pemeriksaan lanjutan (tes dahak, CT-Scan)
                - Jangan menunda penanganan medis
                """)
            else:
                st.success("✅ **NORMAL**")
                st.write("""
                Pola tekstur paru berada dalam batas karakteristik normal.
                
                **Catatan:**
                - Hasil ini bukan diagnosis medis final
                - Tetap lakukan pemeriksaan kesehatan rutin
                - Konsultasikan dengan dokter jika ada keluhan
                """)
            
            # Tampilkan metrik teknis
            st.subheader("📈 Parameter Teknis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediksi Kelas", "TB" if pred == 1 else "Normal")
            
            with col2:
                st.metric("Kode Kelas", int(pred))
            
            with col3:
                st.metric("Jarak Hyperplane", f"{dist_score:.4f}")
            
            with st.expander("ℹ️ Penjelasan Parameter", expanded=False):
                st.write("""
                - **Prediksi Kelas**: Hasil klasifikasi (Normal/TB)
                - **Kode Kelas**: 0 = Normal, 1 = TB
                - **Jarak Hyperplane**: Nilai positif menunjukkan kecenderungan TB, 
                  nilai negatif menunjukkan kecenderungan Normal. Semakin jauh dari 0, 
                  semakin yakin model dengan prediksinya.
                """)
    else:
        st.warning("⚠️ Segmentasi gagal. Tidak dapat melakukan prediksi.")

else:
    st.info("ℹ️ Silakan upload citra X-Ray untuk memulai analisis.")

# ==============================================================================
# FOOTER / DISCLAIMER
# ==============================================================================
st.divider()

st.warning("""
**⚠️ DISCLAIMER**

Aplikasi ini dikembangkan untuk **tujuan akademik/penelitian**. 
Hasil prediksi AI **bukan pengganti diagnosis medis profesional**. 
Selalu konsultasikan hasil citra medis dengan Radiolog atau Dokter Spesialis Paru.
""")

st.caption("© 2025 Kelompok 1 PCD - Universitas Gadjah Mada")