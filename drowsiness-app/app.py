import streamlit as st
import cv2
import numpy as np
import os
import time
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from detector import process_image

# ─────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "StayAwake AI",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

EAR_THRESHOLD = 0.25

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0a0a0f; color: #e2e8f0; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #0a0a0f 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #13131f, #1a1a2e);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        transition: all 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(139, 92, 246, 0.5);
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #4338ca);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
        transform: translateY(-1px);
    }
    .card {
        background: linear-gradient(135deg, #13131f, #1a1a2e);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 20px;
        padding: 1.8rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .card:hover {
        border-color: rgba(139, 92, 246, 0.35);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.1);
    }
    .card-title {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7c3aed;
        margin-bottom: 0.8rem;
    }
    .hero {
        background: linear-gradient(135deg, #13131f 0%, #1a0a2e 50%, #0a1628 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 24px;
        padding: 3.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero h1 {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero p { color: #94a3b8; font-size: 1rem; margin: 0; line-height: 1.7; }
    .hero-badge {
        display: inline-block;
        background: rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.3);
        color: #a78bfa;
        padding: 0.3rem 1rem;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
    }
    .status-normal {
        background: linear-gradient(135deg, #052e16, #064e3b);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        font-size: 1rem;
        font-weight: 600;
        color: #6ee7b7;
    }
    .status-drowsy {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        font-size: 1rem;
        font-weight: 600;
        color: #fca5a5;
        animation: pulse-red 1.5s ease-in-out infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
        50%       { box-shadow: 0 0 20px 4px rgba(239,68,68,0.2); }
    }
    .confidence-bar-wrap {
        background: #1e1e2e;
        border-radius: 100px;
        height: 10px;
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 0.5s ease;
    }
    hr { border-color: rgba(139, 92, 246, 0.1) !important; }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.2rem;
        letter-spacing: -0.01em;
    }
    .section-sub { font-size: 0.85rem; color: #64748b; margin-bottom: 1.5rem; }
    .sidebar-brand {
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(139, 92, 246, 0.15);
        margin-bottom: 1.5rem;
    }
    .sidebar-brand h2 {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sidebar-brand p {
        font-size: 0.7rem;
        color: #64748b !important;
        margin: 0.2rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .tag {
        display: inline-block;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.25);
        color: #a78bfa;
        padding: 0.2rem 0.7rem;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 500;
        margin: 0.15rem;
    }
    .spec-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(139,92,246,0.08);
        font-size: 0.78rem;
    }
    .step-card {
        background: linear-gradient(135deg, #13131f, #1a1a2e);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
defaults = {
    "total_session":     0,
    "total_drowsy":      0,
    "ear_history":       [],
    "confidence_history":[],
    "history":           [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_confidence(ear):
    if ear == 0:
        return 0.0
    if ear < EAR_THRESHOLD:
        conf = min(1.0, (EAR_THRESHOLD - ear) / EAR_THRESHOLD * 2)
    else:
        conf = min(1.0, (ear - EAR_THRESHOLD) / EAR_THRESHOLD * 2)
    return round(conf, 4)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-brand'>
        <h2>StayAwake AI</h2>
        <p>Drowsiness Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "Home",
        "Live Detection",
        "Dashboard Analytics",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='card-title'>Technology Stack</div>
        <span class='tag'>MediaPipe</span>
        <span class='tag'>EAR Method</span>
        <span class='tag'>OpenCV</span>
        <span class='tag'>Streamlit</span>
        <span class='tag'>Python 3.11</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='card-title'>Developer</div>
        <div style='font-size:0.88rem; color:#e2e8f0; font-weight:600;'>Nama Mahasiswa</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:0.3rem;'>NIM</div>
        <div style='font-size:0.75rem; color:#64748b;'>Program Studi Informatika</div>
        <div style='font-size:0.75rem; color:#64748b;'>Image Processing 2025/2026</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# HOME
# ─────────────────────────────────────────
if page == "Home":
    st.markdown("""
    <div class='hero'>
        <img src='assets/logo.png' alt='Logo' style='width:80px; margin-bottom:1.5rem;'>
        <div class='hero-badge'>AI-Powered Drowsiness Detection</div>
        <h1>StayAwake AI</h1>
        <p>Sistem deteksi kantuk berbasis kecerdasan buatan yang dirancang<br>
        untuk membantu mahasiswa tetap fokus saat mengerjakan tugas.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features = [
        ("Live Detection",     "Deteksi kantuk real-time via webcam dengan alarm otomatis"),
        ("EAR Technology",     "Eye Aspect Ratio untuk mengukur kondisi mata secara presisi"),
        ("Dashboard Analytics","Visualisasi statistik dan riwayat deteksi secara lengkap"),
    ]
    for col, (title, desc) in zip([c1, c2, c3], features):
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center; min-height:120px;'>
                <div style='font-size:0.95rem; font-weight:600; color:#e2e8f0;
                            margin-bottom:0.5rem;'>{title}</div>
                <div style='font-size:0.78rem; color:#64748b; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-title'>How It Works</div>
    <div class='section-sub'>Cara kerja sistem deteksi kantuk StayAwake AI</div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("01", "Aktivasi",   "Buka halaman Live Detection dan klik Start"),
        ("02", "Deteksi",    "MediaPipe mendeteksi 478 landmark wajah secara akurat"),
        ("03", "Kalkulasi",  "Sistem menghitung Eye Aspect Ratio dari koordinat mata"),
        ("04", "Alert",      "Alarm berbunyi dan visual alert muncul saat kantuk terdeteksi"),
    ]
    for col, (num, title, desc) in zip([s1, s2, s3, s4], steps):
        with col:
            st.markdown(f"""
            <div class='step-card'>
                <div style='font-size:1.8rem; font-weight:800;
                            background:linear-gradient(135deg,#7c3aed,#4f46e5);
                            -webkit-background-clip:text;
                            -webkit-text-fill-color:transparent;
                            margin-bottom:0.5rem;'>{num}</div>
                <div style='font-size:0.88rem; font-weight:600; color:#e2e8f0;
                            margin-bottom:0.4rem;'>{title}</div>
                <div style='font-size:0.76rem; color:#64748b; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-title'>Technical Specification</div>
    <div class='section-sub'>Parameter dan konfigurasi sistem</div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)
    specs = [
        ("Detection Model", [
            ("Engine",       "MediaPipe Face Mesh"),
            ("Landmarks",    "478 titik wajah"),
            ("Max Faces",    "1 wajah per frame"),
            ("Acceleration", "XNNPACK CPU"),
        ]),
        ("EAR Configuration", [
            ("EAR Threshold",   "0.25"),
            ("Frame Threshold", "15 frame"),
            ("Skip Frame",      "Setiap 2 frame"),
            ("Method",          "Eye Aspect Ratio"),
        ]),
        ("System", [
            ("Platform",  "Streamlit Web App"),
            ("Language",  "Python 3.11"),
            ("Input Mode","Live Webcam"),
            ("Alert",     "Visual + Beep Alarm"),
        ]),
    ]
    for col, (title, items) in zip([t1, t2, t3], specs):
        with col:
            rows = "".join([f"""
            <div class='spec-row'>
                <span style='color:#64748b;'>{k}</span>
                <span style='color:#a78bfa; font-weight:500;'>{v}</span>
            </div>""" for k, v in items])
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>{title}</div>
                {rows}
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# LIVE DETECTION
# ─────────────────────────────────────────
elif page == "Live Detection":
    st.markdown("""
    <div class='section-title'>Live Detection</div>
    <div class='section-sub'>Real-time drowsiness detection menggunakan webcam</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Panduan Penggunaan</div>",
                    unsafe_allow_html=True)

        steps_live = [
            ("1", "Klik tombol Start Live Detection di bawah"),
            ("2", "Window kamera OpenCV akan terbuka secara otomatis"),
            ("3", "Sistem mendeteksi kantuk dan membunyikan alarm beep otomatis"),
            ("4", "Tekan Q pada window kamera untuk menutup sesi"),
        ]
        for num, text in steps_live:
            st.markdown(f"""
            <div style='display:flex; align-items:flex-start; gap:0.8rem; margin-bottom:0.8rem;'>
                <div style='background:rgba(124,58,237,0.15); border-radius:8px;
                            padding:0.25rem 0.65rem; font-size:0.72rem; font-weight:700;
                            color:#a78bfa; min-width:22px; text-align:center;
                            flex-shrink:0;'>{num}</div>
                <div style='font-size:0.83rem; color:#94a3b8; line-height:1.5;'>{text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Start Live Detection", use_container_width=True):
            live_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live.py")
            subprocess.Popen([sys.executable, live_path])
            st.success("Live detection berhasil dimulai. Cek window kamera di taskbar.")

    with col2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Alert System</div>
            <div style='display:flex; flex-direction:column; gap:0.7rem; margin-top:0.5rem;'>
                <div style='background:rgba(239,68,68,0.07);
                            border:1px solid rgba(239,68,68,0.18);
                            border-radius:12px; padding:0.9rem;'>
                    <div style='font-size:0.78rem; font-weight:600; color:#f87171;
                                margin-bottom:0.3rem;'>Alarm Beep</div>
                    <div style='font-size:0.74rem; color:#94a3b8;'>
                        Bunyi alarm 1000Hz berbunyi otomatis saat kantuk terdeteksi</div>
                </div>
                <div style='background:rgba(239,68,68,0.07);
                            border:1px solid rgba(239,68,68,0.18);
                            border-radius:12px; padding:0.9rem;'>
                    <div style='font-size:0.78rem; font-weight:600; color:#f87171;
                                margin-bottom:0.3rem;'>Visual Alert</div>
                    <div style='font-size:0.74rem; color:#94a3b8;'>
                        Overlay merah dan teks peringatan muncul di layar kamera</div>
                </div>
                <div style='background:rgba(16,185,129,0.07);
                            border:1px solid rgba(16,185,129,0.18);
                            border-radius:12px; padding:0.9rem;'>
                    <div style='font-size:0.78rem; font-weight:600; color:#6ee7b7;
                                margin-bottom:0.3rem;'>Auto Reset</div>
                    <div style='font-size:0.74rem; color:#94a3b8;'>
                        Alarm berhenti otomatis ketika mata kembali terbuka</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <div class='card-title'>Tips Optimal</div>
            <div style='font-size:0.78rem; color:#94a3b8; line-height:1.9; margin-top:0.3rem;'>
                Pastikan pencahayaan cukup terang<br>
                Posisi wajah menghadap kamera secara frontal<br>
                Jarak ideal 30 hingga 60 cm dari kamera<br>
                Hindari cahaya latar yang terlalu terang
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# DASHBOARD ANALYTICS
# ─────────────────────────────────────────
elif page == "Dashboard Analytics":
    st.markdown("""
    <div class='section-title'>Dashboard Analytics</div>
    <div class='section-sub'>Statistik dan visualisasi data deteksi sesi live</div>
    """, unsafe_allow_html=True)

    # Baca file log dari live.py
    LOG_FILE = "session_log.csv"

    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
    else:
        df_log = pd.DataFrame(columns=["timestamp","status","ear","confidence"])

    total   = len(df_log)
    drowsy  = len(df_log[df_log["status"] == "MENGANTUK"]) if total > 0 else 0
    normal  = total - drowsy
    acc     = round(normal / total * 100, 1) if total > 0 else 0
    avg_ear = round(df_log["ear"].mean(), 4) if total > 0 else 0
    avg_conf= round(df_log["confidence"].mean() * 100, 1) if total > 0 else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Frame",     total)
    m2.metric("Normal",          normal)
    m3.metric("Mengantuk",       drowsy)
    m4.metric("Akurasi Normal",  f"{acc}%")
    m5.metric("Avg EAR",         avg_ear)
    m6.metric("Avg Confidence",  f"{avg_conf}%")

    st.markdown("<br>", unsafe_allow_html=True)

    if total == 0:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3.5rem;'>
            <div style='font-size:0.95rem; font-weight:600; color:#e2e8f0;
                        margin-bottom:0.4rem;'>Belum Ada Data</div>
            <div style='font-size:0.82rem; color:#64748b;'>
                Jalankan Live Detection terlebih dahulu agar data tersimpan</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)

        # Pie Chart
        with col1:
            st.markdown("<div class='card-title'>Detection Distribution</div>",
                        unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(4, 4), facecolor='#13131f')
            ax1.set_facecolor('#13131f')
            sizes  = [normal if normal > 0 else 0, drowsy if drowsy > 0 else 0]
            colors = ['#10b981', '#ef4444']
            labels = ['Normal', 'Drowsy']
            if sum(sizes) > 0:
                wedges, texts, autotexts = ax1.pie(
                    sizes, labels=labels, colors=colors,
                    autopct='%1.1f%%', startangle=90,
                    textprops={'color': '#94a3b8', 'fontsize': 9},
                    wedgeprops={'linewidth': 2, 'edgecolor': '#13131f'}
                )
                for at in autotexts:
                    at.set_color('#e2e8f0')
                    at.set_fontweight('bold')
            ax1.set_title('Normal vs Drowsy', color='#94a3b8', fontsize=10, pad=15)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

        # EAR History
        with col2:
            st.markdown("<div class='card-title'>EAR Value History</div>",
                        unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor='#13131f')
            ax2.set_facecolor('#13131f')
            x = range(1, len(df_log) + 1)
            ax2.plot(x, df_log["ear"], color='#a78bfa', linewidth=1.5,
                     marker='o', markersize=3, markerfacecolor='#7c3aed')
            ax2.axhline(y=0.25, color='#ef4444', linestyle='--',
                        linewidth=1.2, label='Threshold (0.25)')
            ax2.fill_between(x, df_log["ear"], 0.25,
                             where=[e < 0.25 for e in df_log["ear"]],
                             color='#ef4444', alpha=0.12)
            ax2.set_xlabel('Frame', color='#64748b', fontsize=8)
            ax2.set_ylabel('EAR Value', color='#64748b', fontsize=8)
            ax2.set_title('EAR Value per Frame', color='#94a3b8', fontsize=10, pad=15)
            ax2.tick_params(colors='#64748b', labelsize=7)
            for spine in ['bottom', 'left']:
                ax2.spines[spine].set_color('#334155')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.legend(fontsize=7, facecolor='#1a1a2e',
                       edgecolor='#334155', labelcolor='#94a3b8')
            ax2.grid(axis='y', color='#1e293b', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence Bar Chart
        st.markdown("<div class='card-title'>Confidence Score History</div>",
                    unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(10, 3), facecolor='#13131f')
        ax3.set_facecolor('#13131f')
        x    = range(1, len(df_log) + 1)
        conf = df_log["confidence"] * 100
        ax3.bar(x, conf, color=[
            '#10b981' if c >= 50 else '#ef4444' for c in conf
        ], alpha=0.85, width=0.6)
        ax3.set_xlabel('Frame', color='#64748b', fontsize=8)
        ax3.set_ylabel('Confidence (%)', color='#64748b', fontsize=8)
        ax3.set_title('Confidence Score per Frame', color='#94a3b8', fontsize=10, pad=15)
        ax3.tick_params(colors='#64748b', labelsize=7)
        for spine in ['bottom', 'left']:
            ax3.spines[spine].set_color('#334155')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_ylim(0, 110)
        ax3.grid(axis='y', color='#1e293b', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabel Riwayat
        st.markdown("<div class='card-title'>Detection History</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            df_log.tail(50),
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp":  st.column_config.TextColumn("Timestamp"),
                "status":     st.column_config.TextColumn("Status"),
                "ear":        st.column_config.NumberColumn("EAR Value", format="%.4f"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.2%"),
            }
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Reset Dashboard", use_container_width=False):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        st.rerun()