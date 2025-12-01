import streamlit as st
import numpy as np
import cv2
import joblib
import time
from PIL import Image

# --- IMPORT MODUL FITUR ---
# Pastikan file tb_features.py ada di folder yang sama
try:
    import tb_features as fp
except ImportError:
    st.error("⚠️ Modul 'tb_features.py' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="TBScan Pro - AI Medical Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (TEMA: MODERN MEDICAL SAAS) ---
st.markdown("""
<style>
    /* IMPORT FONT KEREN */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* 1. RESET & BASIC */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stApp {
        background-color: #F1F5F9; /* Slate-100 */
        color: #1E293B;
    }
    
    /* HILANGKAN ELEMENT BAWAAN STREAMLIT YANG MENGGANGGU */
    section[data-testid="stSidebar"] { display: none; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* 2. HEADER */
    .header-container {
        text-align: center;
        padding-bottom: 20px;
        margin-bottom: 20px;
        border-bottom: 1px solid #E2E8F0;
    }
    .header-title {
        color: #0F172A;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    .header-subtitle {
        color: #64748B;
        font-size: 1rem;
        font-weight: 500;
    }

    /* 3. UPLOAD AREA */
    div[data-testid="stFileUploader"] {
        background-color: white;
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        padding: 40px 20px;
        transition: all 0.3s ease;
        text-align: center;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #3B82F6;
        background-color: #F8FAFC;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }
    div[data-testid="stFileUploader"] section {
        padding: 0;
    }

    /* 4. RESULT CARD (DIAGNOSIS) */
    .diagnosis-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    .diagnosis-card:hover {
        transform: translateY(-5px);
    }
    
    /* Style Normal */
    .bg-normal {
        background: linear-gradient(135deg, #ECFDF5 0%, #FFFFFF 100%);
        border: 1px solid #10B981;
    }
    .text-normal-title { color: #059669; font-weight: 800; font-size: 1.8rem; }
    .icon-normal { font-size: 3rem; margin-bottom: 10px; display: block; }

    /* Style TB Positive */
    .bg-positive {
        background: linear-gradient(135deg, #FEF2F2 0%, #FFFFFF 100%);
        border: 1px solid #EF4444;
    }
    .text-positive-title { color: #DC2626; font-weight: 800; font-size: 1.8rem; }
    .icon-positive { font-size: 3rem; margin-bottom: 10px; display: block; }

    /* 5. METRIC BOXES */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="metric-container"] label {
        font-size: 0.85rem;
        color: #64748B;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: #0F172A;
        font-weight: 700;
    }

    /* 6. BUTTONS */
    div.stButton > button {
        background: linear-gradient(to right, #2563EB, #3B82F6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
        color: white;
    }

    /* 7. IMAGE CAPTIONS */
    .caption-text {
        text-align: center;
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_pipeline():
    try:
        pipe = joblib.load("tb_svm_pipeline.joblib")
        return pipe["scaler"], pipe["pca"], pipe["svm"]
    except FileNotFoundError:
        return None, None, None

scaler, pca, svm_clf = load_pipeline()

# --- HEADER SECTION ---
st.markdown("""
<div class="header-container">
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px;">
        <span style="font-size: 3rem;">🫁</span>
        <div>
            <h1 class="header-title">TBScan <span style="color:#3B82F6;">Pro</span></h1>
        </div>
    </div>
    <p class="header-subtitle">Intelligent Tuberculosis Screening System (SVM + GLCM)</p>
</div>
""", unsafe_allow_html=True)

# --- CEK MODEL ---
if scaler is None:
    st.error("❌ **System Error:** Model `tb_svm_pipeline.joblib` tidak ditemukan. Silakan upload file model.")
    st.stop()

# --- LAYOUT GRID UTAMA ---
# Menggunakan Container untuk grouping yang lebih baik
main_container = st.container()

# Inisialisasi variabel seg agar tidak error saat diakses di col_process
seg = None

with main_container:
    col_input, col_process = st.columns([1.2, 1], gap="large")

    # === KOLOM KIRI: INPUT & PREVIEW ===
    with col_input:
        st.markdown("##### 1. Upload Citra X-Ray")
        uploaded = st.file_uploader("Drop X-Ray image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded:
            # Baca & Konversi
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Container putih untuk preview
            with st.container():
                st.markdown("<div style='background:white; padding:15px; border-radius:12px; border:1px solid #E2E8F0; margin-top:15px;'>", unsafe_allow_html=True)
                
                tab1, tab2 = st.tabs(["🖼️ Citra Asli", "🎯 Segmentasi ROI"])
                
                with tab1:
                    st.image(img_gray, use_container_width=True)
                    st.markdown(f"<p class='caption-text'>Resolusi: {img_gray.shape[1]}x{img_gray.shape[0]} px</p>", unsafe_allow_html=True)
                
                # Proses Segmentasi Otomatis
                seg = None
                try:
                    with st.spinner("🔄 Memproses segmentasi paru..."):
                        pre = fp.preprocess_image(img_gray)
                        seg, mask = fp.segment_lungs_final(pre)
                    
                    with tab2:
                        st.image(seg, use_container_width=True)
                        st.markdown("<p class='caption-text'>Area Paru Tersegmentasi</p>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Gagal segmentasi: {e}")
                
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Placeholder State (Jika belum upload)
            st.info("ℹ️ Silakan unggah citra rontgen dada (Chest X-Ray) pada kotak di atas untuk memulai analisis.")

    # === KOLOM KANAN: HASIL DIAGNOSIS ===
    with col_process:
        st.markdown("##### 2. Analisis AI")
        # Tombol Analisis (Selalu Muncul)
        if st.button("🚀 Jalankan Deteksi", type="primary", use_container_width=True):
            if not uploaded:
                st.warning("⚠️ Silakan unggah citra X-Ray pada panel kiri terlebih dahulu sebelum memulai deteksi.")
            elif seg is None:
                st.error("⚠️ Gagal memproses segmentasi citra. Pastikan citra yang diunggah valid.")
            else:
                # Progress Bar Buatan agar terlihat interaktif
                progress_text = "Menganalisis pola tekstur..."
                my_bar = st.progress(0, text=progress_text)
                
                for percent_complete in range(0, 100, 20):
                    time.sleep(0.05)
                    my_bar.progress(percent_complete + 20, text=progress_text)
                
                # --- CORE LOGIC ---
                feats = fp.glcm_stat_features(seg)
                feats = feats.reshape(1, -1)
                
                X_std = scaler.transform(feats)
                X_pca = pca.transform(X_std)
                pred = svm_clf.predict(X_pca)[0]
                
                dist_score = 0.0
                if hasattr(svm_clf, "decision_function"):
                    dist_score = svm_clf.decision_function(X_pca)[0]
                
                my_bar.empty() # Hapus progress bar

                # --- UI HASIL DIAGNOSIS ---
                is_tb = (pred == 1)
                
                # Setup Variabel Tampilan
                if is_tb:
                    css_class = "bg-positive"
                    title_class = "text-positive-title"
                    icon = "⚠️"
                    label = "TERINDIKASI TB"
                    desc = "Sistem mendeteksi pola tekstur abnormal yang diasosiasikan dengan Tuberculosis."
                else:
                    css_class = "bg-normal"
                    title_class = "text-normal-title"
                    icon = "✅"
                    label = "NORMAL"
                    desc = "Pola tekstur paru berada dalam batas karakteristik normal."

                # Render Kartu Hasil
                st.markdown(f"""
                <div class="diagnosis-card {css_class}">
                    <span class="icon-{ 'positive' if is_tb else 'normal' }">{icon}</span>
                    <div style="font-size: 14px; text-transform:uppercase; letter-spacing:1px; color:#64748B; margin-bottom:5px;">Hasil Prediksi</div>
                    <div class="{title_class}">{label}</div>
                    <hr style="border: 0; border-top: 1px solid rgba(0,0,0,0.1); margin: 20px 0;">
                    <div style="color: #475569; font-size: 15px; line-height: 1.5;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

                # --- METRIK TEKNIS ---
                st.markdown("###### Parameter Deteksi")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Prediksi Kelas", f"{int(pred)}")
                with m2:
                    st.metric("Jarak Hyperplane", f"{dist_score:.4f}")
                
# --- FOOTER / DISCLAIMER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="
        text-align: center; 
        padding: 20px;
        background-color: #F8FAFC;
        border-top: 1px solid #E2E8F0;
        color: #64748B;
        font-size: 12px;
    ">
        <p style="font-weight: 600; color: #475569; margin-bottom: 5px;">⚠️ DISCLAIMER</p>
        <p style="max-width: 600px; margin: 0 auto; line-height: 1.6;">
            Aplikasi ini dikembangkan untuk <b>Pemenuhan Proyek Akademik</b>. 
            Hasil prediksi AI <u>bukan pengganti diagnosis medis profesional</u>. 
            Selalu konsultasikan hasil citra medis dengan Radiolog atau Dokter Spesialis Paru.
        </p>
        <p style="margin-top: 15px; opacity: 0.7;">© 2024 TBScan Pro Project - Powered by Python & OpenCV</p>
    </div>
""", unsafe_allow_html=True)