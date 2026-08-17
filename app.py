import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Super Six | Solar Vision AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# FILES
# ============================================================

CSV_FILE = "inspection_history.csv"
IMAGE_DIR = "saved_images"
LOGO_FILE = "logo.png"

MODEL_FILES = {
    "dust": "dust_model.pt",
    "crack": "crack_model.pt",
    "hotspot": "hotspot_model.pt",
}

# ── Estimation constants (single source of truth) ─────────────
DUST_LOSS_PER    = 2.5    # % loss per detected dust region
DUST_LOSS_MAX    = 30.0   # cap
CRACK_LOSS_PER   = 5.0
CRACK_LOSS_MAX   = 40.0
HOTSPOT_LOSS_PER = 15.0
HOTSPOT_LOSS_MAX = 60.0

# Health-score penalty per detection (weighted by severity)
DUST_HEALTH_PEN    = 3
CRACK_HEALTH_PEN   = 8
HOTSPOT_HEALTH_PEN = 20

MAX_UPLOAD_MB = 15        # Reject images larger than this


# ============================================================
# DARK FUTURISTIC CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: Inter, Arial, sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 8% 10%,
            rgba(0, 229, 255, 0.10),
            transparent 25%
        ),
        radial-gradient(
            circle at 92% 12%,
            rgba(255, 176, 0, 0.09),
            transparent 24%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(0, 255, 170, 0.06),
            transparent 30%
        ),
        #02050A;

    color: #F5F7FA;
}

/* Background grid */

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;

    background-image:
        linear-gradient(
            rgba(0, 229, 255, 0.025) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(0, 229, 255, 0.025) 1px,
            transparent 1px
        );

    background-size: 36px 36px;

    pointer-events: none;
    z-index: 0;
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}

[data-testid="stDecoration"] {
    display: none;
}

/* Main container */

.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 4rem;
    position: relative;
    z-index: 1;
}


/* ============================================================
   SUPER SIX BRAND HEADER
   ============================================================ */

.super-header {
    min-height: 96px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 18px;
    margin-bottom: 8px;
    border: 1px solid #12334B;
    border-radius: 20px;
    background:
        radial-gradient(circle at 12% 50%, rgba(0,229,255,.10), transparent 25%),
        linear-gradient(135deg, rgba(4,14,24,.98), rgba(2,7,13,.98));
    box-shadow: 0 12px 40px rgba(0,0,0,.28), inset 0 0 35px rgba(0,229,255,.025);
}
.super-brand-wrap { display:flex; align-items:center; gap:14px; }
.super-brand-copy { display:flex; flex-direction:column; }
.super-brand-title { font-size:26px; line-height:1; font-weight:950; letter-spacing:1.2px; color:#F4F8FF; }
.super-brand-title span { color:#00E5FF; text-shadow:0 0 18px rgba(0,229,255,.35); }
.super-brand-subtitle { margin-top:6px; color:#6F879D; font-size:10px; letter-spacing:3px; font-weight:800; }
.super-header-right { display:flex; align-items:center; gap:16px; }
.sun-orb {
    width:42px; height:42px; display:flex; align-items:center; justify-content:center;
    border-radius:50%; border:1px solid rgba(255,176,0,.45); color:#FFB000; font-size:22px;
    box-shadow:0 0 24px rgba(255,176,0,.16); animation:sunPulse 3s ease-in-out infinite;
}
@keyframes sunPulse {
    0%,100% { box-shadow:0 0 18px rgba(255,176,0,.12); }
    50% { box-shadow:0 0 32px rgba(255,176,0,.32); }
}
.top-marker {
    display:flex; justify-content:space-between; align-items:center;
    padding:8px 4px 13px; border-bottom:1px solid #102338; margin-bottom:14px;
}
.top-marker-left { color:#6E8095; font-size:10px; letter-spacing:2px; font-weight:800; }
.top-marker-right { color:#00FFB0; font-size:10px; font-weight:800; }
.logo-fallback {
    width:74px; height:74px; border-radius:18px; display:flex; align-items:center;
    justify-content:center; font-size:35px; border:1px solid #00E5FF; background:#06131F;
}
.visual-image-card {
    border:1px solid #12435C; border-radius:16px; padding:6px;
    background:linear-gradient(145deg,#07121E,#03080E);
    box-shadow:0 0 25px rgba(0,229,255,.06); overflow:hidden; min-height:0;
}
.visual-image-card img { width:100% !important; border-radius:11px; object-fit:contain; }
[data-testid="stImage"] img { border-radius:14px !important; border:1px solid #12435C; }

/* ============================================================
   TOP HEADER
   ============================================================ */

.top-header {
    min-height: 86px;

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 12px 20px;

    border: 1px solid #172538;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(8, 16, 27, 0.96),
            rgba(3, 7, 13, 0.96)
        );

    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.35);

    margin-bottom: 22px;
}

.brand-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 1px;
    color: white;
}

.brand-subtitle {
    font-size: 11px;
    color: #728096;
    letter-spacing: 2px;
    margin-top: 3px;
}

.online-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 9px 15px;

    border-radius: 999px;

    color: #00FFB0;

    border: 1px solid rgba(0,255,176,0.35);

    background: rgba(0,255,176,0.05);

    font-size: 12px;
    font-weight: 700;
}

.online-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #00FFB0;

    box-shadow: 0 0 12px #00FFB0;

    animation: pulseDot 1.5s infinite;
}

@keyframes pulseDot {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }

    50% {
        opacity: 0.35;
        transform: scale(0.7);
    }
}


/* ============================================================
   HERO
   ============================================================ */

.hero-card {
    position: relative;
    overflow: hidden;

    padding: 35px 25px;

    border-radius: 24px;

    border: 1px solid rgba(0,229,255,0.22);

    background:
        linear-gradient(
            135deg,
            rgba(0,229,255,0.055),
            rgba(255,176,0,0.035),
            rgba(5,9,16,0.96)
        );

    text-align: center;

    margin-bottom: 24px;
}

.hero-card::after {
    content: "";

    position: absolute;

    width: 320px;
    height: 320px;

    border-radius: 50%;

    background: rgba(255,176,0,0.12);

    filter: blur(80px);

    right: -120px;
    top: -180px;

    animation: sunGlow 5s ease-in-out infinite;
}

@keyframes sunGlow {
    0%, 100% {
        opacity: 0.35;
        transform: scale(0.9);
    }

    50% {
        opacity: 0.8;
        transform: scale(1.1);
    }
}

.hero-title {
    font-size: clamp(32px, 5vw, 60px);

    font-weight: 900;

    letter-spacing: -2px;

    background:
        linear-gradient(
            90deg,
            #FFFFFF,
            #00E5FF,
            #FFFFFF,
            #FFB000
        );

    background-size: 300% auto;

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: gradientMove 6s linear infinite;
}

@keyframes gradientMove {
    0% {
        background-position: 0% center;
    }

    100% {
        background-position: 300% center;
    }
}

.hero-subtitle {
    color: #8995A7;
    font-size: 15px;
    margin-top: 8px;
}

.hero-pills {
    margin-top: 20px;

    display: flex;
    justify-content: center;

    gap: 10px;

    flex-wrap: wrap;
}

.hero-pill {
    padding: 7px 14px;

    border-radius: 999px;

    border: 1px solid #20344A;

    background: rgba(255,255,255,0.025);

    color: #9AA7B8;

    font-size: 11px;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {
    font-size: 29px;
    font-weight: 800;
    color: white;
    margin-top: 12px;
}

.section-subtitle {
    color: #768397;
    margin-bottom: 22px;
}


/* ============================================================
   TABS
   ============================================================ */

.stTabs [data-baseweb="tab-list"] {
    background: rgba(5,10,17,0.88);

    border: 1px solid #172538;

    border-radius: 15px;

    padding: 5px;

    gap: 5px;

    margin-bottom: 25px;
}

.stTabs [data-baseweb="tab"] {
    color: #7D899B;

    border-radius: 10px;

    padding: 11px 20px;

    font-weight: 700;
}

.stTabs [data-baseweb="tab"]:hover {
    color: white;
    background: rgba(0,229,255,0.05);
}

.stTabs [aria-selected="true"] {
    color: white !important;

    background:
        linear-gradient(
            135deg,
            rgba(0,229,255,0.14),
            rgba(255,176,0,0.08)
        );

    border-bottom: 2px solid #00E5FF;
}


/* ============================================================
   CONTROL CARD
   ============================================================ */

.control-card {
    background:
        linear-gradient(
            145deg,
            rgba(10,18,30,0.98),
            rgba(4,8,14,0.98)
        );

    border: 1px solid #1B2A3C;

    border-radius: 20px;

    padding: 22px;

    margin-bottom: 22px;
}

.stSelectbox > div > div {
    background: #080F18 !important;

    color: white !important;

    border: 1px solid #24364B !important;

    border-radius: 11px !important;
}

[data-testid="stFileUploader"] {
    background:
        linear-gradient(
            135deg,
            rgba(0,229,255,0.035),
            rgba(255,176,0,0.025)
        );

    border: 1px dashed rgba(0,229,255,0.45);

    border-radius: 15px;

    padding: 10px;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    min-height: 48px;

    border-radius: 12px !important;

    border: 1px solid #00E5FF !important;

    background:
        linear-gradient(
            135deg,
            #009FEF,
            #0066FF
        ) !important;

    color: white !important;

    font-weight: 800 !important;

    box-shadow:
        0 0 25px rgba(0,229,255,0.15);

    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 0 35px rgba(0,229,255,0.35);

    border-color: white !important;
}


/* ============================================================
   IMAGE CARD
   ============================================================ */

.image-title {
    color: #AEB8C7;

    font-size: 13px;

    font-weight: 800;

    letter-spacing: 0.7px;

    margin-bottom: 8px;
}

.image-frame {
    background: #050A11;

    border: 1px solid #1D3044;

    border-radius: 17px;

    padding: 8px;

    overflow: hidden;
}


/* ============================================================
   METRICS
   ============================================================ */

.metric-grid {
    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 14px;

    margin-top: 20px;
}

.metric-card {
    position: relative;

    overflow: hidden;

    background:
        linear-gradient(
            145deg,
            #0D1723,
            #060A10
        );

    border: 1px solid #1C2C40;

    border-radius: 16px;

    padding: 24px;

    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);

    border-color: rgba(0,229,255,0.4);
}

.metric-label {
    color: #78869A;

    font-size: 11px;

    letter-spacing: 1px;

    text-transform: uppercase;
}

.metric-value {
    font-size: 40px;

    font-weight: 900;

    margin-top: 6px;

    color: white;
}

.metric-small {
    color: #59677A;

    font-size: 11px;

    margin-top: 3px;
}


/* ============================================================
   REPORT CARDS
   ============================================================ */

.report-card {
    background:
        linear-gradient(
            145deg,
            rgba(10,17,28,0.98),
            rgba(4,8,14,0.98)
        );

    border: 1px solid #1D2B3D;

    border-radius: 18px;

    padding: 22px;

    margin-top: 18px;
}

.good {
    border-left: 4px solid #00FFB0;
}

.warning {
    border-left: 4px solid #FFB000;
}

.danger {
    border-left: 4px solid #FF4757;
}


/* ============================================================
   HEALTH
   ============================================================ */

.health-box {
    text-align: center;

    padding: 10px;
}

.health-circle {
    width: 150px;
    height: 150px;

    margin: 15px auto;

    border-radius: 50%;

    display: flex;

    align-items: center;
    justify-content: center;

    background:
        conic-gradient(
            #00FFB0 0deg,
            #00E5FF 180deg,
            #FFB000 290deg,
            #FF4757 360deg
        );

    box-shadow:
        0 0 35px rgba(0,229,255,0.18);
}

.health-inner {
    width: 120px;
    height: 120px;

    border-radius: 50%;

    background: #070C13;

    display: flex;

    align-items: center;
    justify-content: center;

    flex-direction: column;
}

.health-score {
    font-size: 32px;
    font-weight: 900;
}

.health-label {
    font-size: 10px;
    color: #738096;
}


/* ============================================================
   ABOUT
   ============================================================ */

.about-card {
    background:
        linear-gradient(
            145deg,
            rgba(10,18,30,0.98),
            rgba(4,8,14,0.98)
        );

    border: 1px solid #1D2B3D;

    border-radius: 20px;

    padding: 25px;

    margin-bottom: 18px;
}

.team-grid {
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 12px;

    margin-top: 15px;
}

.team-member {
    text-align: center;

    padding: 16px 10px;

    border-radius: 13px;

    border: 1px solid #203147;

    background: rgba(255,255,255,0.02);

    color: #D9E0EA;

    transition: all 0.25s ease;
}

.team-member:hover {
    transform: translateY(-3px);

    border-color: #00E5FF;

    box-shadow:
        0 0 22px rgba(0,229,255,0.08);
}

.lead-card {
    text-align: center;

    border: 1px solid rgba(255,176,0,0.45);

    background:
        linear-gradient(
            135deg,
            rgba(255,176,0,0.07),
            rgba(0,229,255,0.04)
        );

    border-radius: 18px;

    padding: 20px;

    margin-top: 18px;
}


/* ============================================================
   HISTORY
   ============================================================ */

.history-item {
    background: #080E17;

    border: 1px solid #1C2B3E;

    border-radius: 15px;

    padding: 18px;

    margin-bottom: 12px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    color: #5F6C7E;

    font-size: 12px;

    padding-top: 35px;

    margin-top: 45px;

    border-top: 1px solid #172131;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 900px) {

    .metric-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .team-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 600px) {

    .metric-grid {
        grid-template-columns: 1fr;
    }

    .team-grid {
        grid-template-columns: 1fr;
    }

    .top-header {
        padding: 10px;
    }

    .hero-title {
        font-size: 34px;
    }
}


/* ============================================================
   STREAMLIT SHELL RESET — FORCE FULL DARK VIEWPORT
   ============================================================ */

/* The white strip above the application is Streamlit's header/chrome. */
html,
body,
#root,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.stApp {
    background-color: #02050A !important;
    color: #F5F7FA !important;
}

/* Completely remove the default Streamlit header area. */
header[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    background: #02050A !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* Extra protection for Streamlit's header children. */
header[data-testid="stHeader"] *,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background-color: #02050A !important;
}

/* Make the main app start at the very top. */
[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

[data-testid="stMain"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
    background-color: #02050A !important;
}

section.main {
    background-color: #02050A !important;
}

section.main > div {
    padding-top: 0 !important;
}

/* Remove any accidental white root/background surfaces. */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stColumn"],
[data-testid="stElementContainer"] {
    background: transparent !important;
}

/* Main content spacing — no large blank band at the top. */
.block-container {
    padding-top: 0.65rem !important;
    padding-bottom: 3rem !important;
    max-width: 1450px !important;
}

/* Keep uploader, dropdowns and controls dark too. */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
[data-baseweb="select"] > div,
[data-testid="stTextInput"] > div,
[data-testid="stNumberInput"] > div {
    background-color: #07121D !important;
}

/* Streamlit buttons must remain readable. */
.stButton > button {
    color: #FFFFFF !important;
}

/* Kill the footer/background if Streamlit renders it. */
footer,
[data-testid="stFooter"] {
    background: #02050A !important;
    color: #02050A !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# STORAGE
# ============================================================

def initialize_storage():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    if not os.path.exists(CSV_FILE):
        columns = [
            "Timestamp",
            "Filename",
            "Inspection_Mode",
            "Dust_Status",
            "Crack_Status",
            "Hotspot_Status",
            "Health_Score",
            "Priority",
            "Detections_Count",
            "Estimated_Loss_Pct",
            "Saved_Image_Path",
        ]

        pd.DataFrame(columns=columns).to_csv(
            CSV_FILE,
            index=False
        )


def save_record(record, image_array):

    os.makedirs(IMAGE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_name = (
        str(record["Filename"])
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    image_name = (
        f"{timestamp}_{safe_name}"
    )

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    Image.fromarray(image_array).save(
        image_path
    )

    record["Saved_Image_Path"] = image_path

    # Write header only when the file is empty or brand-new
    _write_header = (
        not os.path.exists(CSV_FILE)
        or os.path.getsize(CSV_FILE) == 0
    )
    pd.DataFrame([record]).to_csv(
        CSV_FILE,
        mode="a",
        header=_write_header,
        index=False,
    )

    st.toast(
        "Analysis saved successfully!",
        icon="✅"
    )


initialize_storage()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    loaded = {}

    for name, filename in MODEL_FILES.items():

        if os.path.exists(filename):

            try:
                loaded[name] = YOLO(filename)

            except Exception as e:

                st.error(
                    f"Unable to load {filename}: {e}"
                )

                loaded[name] = None

        else:

            loaded[name] = None

    return loaded


models = load_models()


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="super-header">
    <div class="super-brand-wrap">
        <div class="super-brand-copy">
            <div class="super-brand-title">SUPER <span>SIX</span></div>
            <div class="super-brand-subtitle">SOLAR VISION AI</div>
        </div>
    </div>
    <div class="super-header-right">
        <div class="online-pill">
            <span class="online-dot"></span>
            AI SYSTEM ONLINE
        </div>
        <div class="sun-orb">☀</div>
    </div>
</div>
""")

# Actual logo in its own centered row so Streamlit cannot clip it.
logo_left, logo_center, logo_right = st.columns([1, 1, 1])
with logo_center:
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=150)
    else:
        st.markdown('<div class="logo-fallback">☀️</div>', unsafe_allow_html=True)

st.html("""
<div style="height:1px;background:linear-gradient(90deg,transparent,#00E5FF,#17324B,#FFB000,transparent);
            margin:0 0 14px 0;"></div>
""")

# ============================================================
# NAVIGATION
# ============================================================
st.html("""
<div class="top-marker">
    <div class="top-marker-left">SUPER SIX • SOLAR VISION AI • HACKATHON</div>
    <div class="top-marker-right">● AI SYSTEM ONLINE</div>
</div>
""")

# TABS
# ============================================================

tab_inspect, tab_history, tab_about = st.tabs(
    [
        "🔬 INSPECT",
        "📁 INSPECTION HISTORY",
        "ℹ️ ABOUT PROJECT",
    ]
)


# ============================================================
# INSPECT
# ============================================================

with tab_inspect:

    # ========================================================
    # HACKATHON INSPECTION CENTER — 3 COLUMN COMMAND DECK
    # ========================================================
    st.html("""
    <div class="section-title">🔬 AI INSPECTION CENTER <span style="font-size:10px;color:#00FFB0;border:1px solid #00FFB0;padding:4px 8px;border-radius:99px;vertical-align:middle;margin-left:8px;">HACKATHON UI V2</span></div>
    <div class="section-subtitle">
        Upload a solar panel image and let Super Six detect soiling,
        structural cracks and thermal hotspots.
    </div>
    """)

    # Controls / visual input / project information
    left_col, center_col, right_col = st.columns(
        [0.72, 2.15, 1.03], gap="large"
    )

    with left_col:
        st.html("""
        <div class="control-card" style="padding:20px;">
            <div style="color:#00E5FF;font-size:13px;font-weight:800;
                        letter-spacing:.5px;margin-bottom:10px;">
                ⚙️ INSPECTION CONTROL
            </div>
        </div>
        """)

        inspection_mode = st.selectbox(
            "Inspection Mode",
            [
                "Dust Detection",
                "Crack Detection",
                "Hotspot Detection",
                "Comprehensive Analysis",
            ],
            key="inspection_mode_main",
        )

        confidence_threshold = st.slider(
            "🎯 AI Confidence Threshold",
            0.05, 1.00, 0.40, 0.05,
            key="confidence_main",
        )

        uploaded_file = st.file_uploader(
            "📤 Upload Solar Panel Image",
            type=["jpg", "jpeg", "png"],
            key="solar_image_upload",
        )

        if inspection_mode == "Hotspot Detection":
            st.warning("🔥 Thermal/infrared imagery is recommended for hotspot detection.")
        elif inspection_mode == "Comprehensive Analysis":
            st.info("🧠 Runs your trained soiling, crack and hotspot models.")

        if uploaded_file is not None:
            st.html(f"""
            <div class="mini-file-card">
                <div style="font-size:11px;color:#6F8299;">INPUT IMAGE</div>
                <div style="font-weight:800;color:#EAF2FF;margin-top:4px;
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                    {uploaded_file.name}
                </div>
            </div>
            """)

        st.html("""
        <div style="margin-top:18px;padding:14px;border-radius:14px;
                    border:1px solid #17324A;background:#06111C;">
            <div style="color:#00FFB0;font-size:12px;font-weight:800;">
                ● AI SYSTEM ONLINE
            </div>
            <div style="color:#74869A;font-size:11px;margin-top:5px;">
                Custom trained YOLO models ready
            </div>
        </div>
        """)

        # Action stays in the control panel, directly below the upload area.
        st.markdown(
            """
            <div style="
                margin-top:14px;
                margin-bottom:6px;
                color:#00E5FF;
                font-size:10px;
                font-weight:900;
                letter-spacing:1px;
            ">ACTION</div>
            """,
            unsafe_allow_html=True,
        )

        run_analysis = st.button(
            "🚀  RUN AI ANALYSIS",
            use_container_width=True,
            key="run_analysis_new_ui",
        )

    # Persistent state must exist before the image/output columns render.
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None

    with center_col:

        # Original and AI output are side-by-side for a cleaner inspection workflow.
        input_view, output_view = st.columns(2, gap="small")

        with input_view:
            st.html('<div class="image-title">📸 ORIGINAL INPUT</div>')

            if uploaded_file is None:
                st.html("""
                <div class="image-frame" style="height:260px;display:flex;
                            align-items:center;justify-content:center;flex-direction:column;">
                    <div style="font-size:52px;">☀️</div>
                    <div style="font-size:15px;font-weight:800;color:#DDE7F3;">
                        Ready for Solar Inspection
                    </div>
                    <div style="color:#728096;font-size:11px;margin-top:7px;text-align:center;">
                        Upload a clear solar-panel image from the control panel.
                    </div>
                </div>
                """)
            else:
                file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
                original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if original_bgr is None:
                    st.error("❌ Unable to read this image.")
                    st.stop()
                original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
                st.image(original_rgb, use_container_width=True)

        with output_view:
            st.html('<div class="image-title">🤖 AI DETECTION OUTPUT</div>')

            if st.session_state.analysis_data is not None:
                st.image(
                    st.session_state.analysis_data["results_img"],
                    use_container_width=True
                )
            else:
                st.html("""
                <div class="image-frame" style="height:260px;display:flex;
                            align-items:center;justify-content:center;flex-direction:column;">
                    <div style="font-size:48px;">🤖</div>
                    <div style="color:#DDE7F3;font-size:14px;font-weight:800;">
                        AI Detection Output
                    </div>
                    <div style="color:#6F8299;font-size:11px;margin-top:7px;text-align:center;">
                        Click RUN AI ANALYSIS in the left control panel.
                    </div>
                </div>
                """)

        # Keep the image area focused. Detailed analysis is rendered below the 3-column deck.
        if st.session_state.analysis_data is not None:
            _d = st.session_state.analysis_data
            st.markdown(
                f"""
                <div style="margin-top:12px;padding:14px 16px;border:1px solid #173E56;
                            border-radius:14px;background:linear-gradient(135deg,#06131E,#081B26);">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
                        <div>
                            <div style="color:#00E5FF;font-size:11px;font-weight:900;letter-spacing:1.2px;">AI ANALYSIS COMPLETE</div>
                            <div style="color:#AAB7C8;font-size:11px;margin-top:4px;">
                                {_d["total_detections"]} detected regions • {_d["priority"]} priority • Avg confidence {_d["avg_confidence"]:.2f}
                            </div>
                        </div>
                        <div style="font-size:26px;font-weight:900;color:#00FFB0;">{_d["health_numeric"]}<span style="font-size:11px;color:#718198;">/100</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="margin-top:12px;padding:11px 14px;border:1px dashed #23415A;
                            border-radius:12px;background:#050B12;color:#6F8299;font-size:11px;text-align:center;">
                    Upload an image and press <b style="color:#00E5FF;">RUN AI ANALYSIS</b> to generate the annotated image and detailed inspection report.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.html("""
        <div class="about-card" style="padding:18px;">
            <div style="font-size:18px;font-weight:900;color:#EAF2FF;margin-bottom:12px;">
                ℹ️ ABOUT SUPER SIX
            </div>
            <p style="color:#AAB7C8;line-height:1.65;font-size:12px;">
                Super Six is an AI-powered computer vision platform designed to
                assist solar-panel inspection and maintenance.
            </p>
            <p style="color:#AAB7C8;line-height:1.65;font-size:12px;">
                Your custom-trained YOLO models identify soiling, cracks and
                thermal hotspot patterns from inspection images.
            </p>
            <hr style="border-color:#193047;">
            <div style="color:#00E5FF;font-size:14px;font-weight:900;margin-bottom:9px;">
                👥 TEAM SUPER SIX
            </div>
            <div class="team-grid" style="grid-template-columns:repeat(2,1fr);gap:7px;">
                <div class="team-member" style="padding:9px;font-size:11px;">Abhijeet Singh</div>
                <div class="team-member" style="padding:9px;font-size:11px;">Nidhi</div>
                <div class="team-member" style="padding:9px;font-size:11px;">Ansh</div>
                <div class="team-member" style="padding:9px;font-size:11px;">Anubhav</div>
                <div class="team-member" style="padding:9px;font-size:11px;">Akanksha</div>
                <div class="team-member" style="padding:9px;font-size:11px;">Trisha</div>
            </div>
            <hr style="border-color:#193047;margin:13px 0;">
            <div style="color:#00E5FF;font-size:14px;font-weight:900;margin-bottom:8px;">
                🛠️ TECHNOLOGY STACK
            </div>
            <div style="color:#C7D1DE;font-size:11px;line-height:1.9;">
                • Python &nbsp; • Streamlit<br>
                • YOLO Object Detection<br>
                • OpenCV &nbsp; • Pandas<br>
                • Custom Trained AI Models
            </div>
            <hr style="border-color:#193047;margin:13px 0;">
            <div style="color:#FFB000;font-size:14px;font-weight:900;">
                👑 LEAD DEVELOPER
            </div>
            <div style="font-size:17px;font-weight:900;color:white;margin-top:5px;">
                Abhijeet Singh
            </div>
        </div>
        """)

    # ========================================================
    # ORIGINAL AI ANALYSIS ENGINE — PRESERVED
# ========================================================
    if run_analysis:

        # ── Guard: image must be uploaded ────────────────────────
        if uploaded_file is None:
            st.error(
                "❌ Please upload a solar panel image before running analysis."
            )
            st.stop()

        # ── Guard: reject oversized files ────────────────────────
        if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(
                f"❌ File too large ({uploaded_file.size / 1024 / 1024:.1f} MB). "
                f"Max allowed size is {MAX_UPLOAD_MB} MB."
            )
            st.stop()

        with st.spinner(
            "🧠 Running Super Six AI models…"
        ):

            output_bgr = original_bgr.copy()

            total_detections = 0

            confidence_values = []
            detection_details = []
            image_h, image_w = original_bgr.shape[:2]

            dust_count = 0
            crack_count = 0
            hotspot_count = 0

            dust_status = "UNTESTED"
            crack_status = "UNTESTED"
            hotspot_status = "UNTESTED"


            # ============================================
            # DUST MODEL
            # ============================================

            if inspection_mode in [
                "Dust Detection",
                "Comprehensive Analysis",
            ]:

                model = models.get("dust")

                if model is not None:

                    result = model(
                        original_bgr,
                        conf=confidence_threshold,
                        verbose=False,
                    )[0]

                    dust_count = len(
                        result.boxes
                    )

                    total_detections += dust_count

                    if dust_count > 0:

                        dust_status = (
                            f"DETECTED ({dust_count} regions)"
                        )

                    else:

                        dust_status = "NONE"

                    for box in result.boxes:

                        confidence = float(
                            box.conf[0]
                        )

                        confidence_values.append(
                            confidence
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist()
                        )

                        region_area_pct = (
                            max(0, x2 - x1) * max(0, y2 - y1)
                            / max(1, image_w * image_h)
                            * 100.0
                        )
                        detection_details.append({
                            "type": "Soiling / Dust",
                            "confidence": confidence,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "area_pct": region_area_pct,
                        })

                        cv2.rectangle(
                            output_bgr,
                            (x1, y1),
                            (x2, y2),
                            (255, 190, 0),
                            3,
                        )

                        label = (
                            f"DUST "
                            f"{confidence:.2f}"
                        )

                        cv2.putText(
                            output_bgr,
                            label,
                            (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 190, 0),
                            2,
                            cv2.LINE_AA,
                        )

                else:

                    dust_status = "MODEL MISSING"


            # ============================================
            # CRACK MODEL
            # ============================================

            if inspection_mode in [
                "Crack Detection",
                "Comprehensive Analysis",
            ]:

                model = models.get("crack")

                if model is not None:

                    result = model(
                        original_bgr,
                        conf=confidence_threshold,
                        verbose=False,
                    )[0]

                    crack_count = len(
                        result.boxes
                    )

                    total_detections += crack_count

                    if crack_count > 0:

                        crack_status = (
                            f"DETECTED ({crack_count} regions)"
                        )

                    else:

                        crack_status = "NONE"

                    for box in result.boxes:

                        confidence = float(
                            box.conf[0]
                        )

                        confidence_values.append(
                            confidence
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist()
                        )

                        region_area_pct = (
                            max(0, x2 - x1) * max(0, y2 - y1)
                            / max(1, image_w * image_h)
                            * 100.0
                        )
                        detection_details.append({
                            "type": "Structural Crack",
                            "confidence": confidence,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "area_pct": region_area_pct,
                        })

                        cv2.rectangle(
                            output_bgr,
                            (x1, y1),
                            (x2, y2),
                            (255, 70, 90),
                            3,
                        )

                        label = (
                            f"CRACK "
                            f"{confidence:.2f}"
                        )

                        cv2.putText(
                            output_bgr,
                            label,
                            (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 70, 90),
                            2,
                            cv2.LINE_AA,
                        )

                else:

                    crack_status = "MODEL MISSING"


            # ============================================
            # HOTSPOT MODEL
            # ============================================

            if inspection_mode in [
                "Hotspot Detection",
                "Comprehensive Analysis",
            ]:

                model = models.get("hotspot")

                if model is not None:

                    result = model(
                        original_bgr,
                        conf=confidence_threshold,
                        verbose=False,
                    )[0]

                    hotspot_count = len(
                        result.boxes
                    )

                    total_detections += hotspot_count

                    if hotspot_count > 0:

                        hotspot_status = (
                            f"DETECTED ({hotspot_count} regions)"
                        )

                    else:

                        hotspot_status = "NONE"

                    for box in result.boxes:

                        confidence = float(
                            box.conf[0]
                        )

                        confidence_values.append(
                            confidence
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist()
                        )

                        region_area_pct = (
                            max(0, x2 - x1) * max(0, y2 - y1)
                            / max(1, image_w * image_h)
                            * 100.0
                        )
                        detection_details.append({
                            "type": "Thermal Hotspot",
                            "confidence": confidence,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "area_pct": region_area_pct,
                        })

                        cv2.rectangle(
                            output_bgr,
                            (x1, y1),
                            (x2, y2),
                            (0, 170, 255),
                            3,
                        )

                        label = (
                            f"HOTSPOT "
                            f"{confidence:.2f}"
                        )

                        cv2.putText(
                            output_bgr,
                            label,
                            (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 170, 255),
                            2,
                            cv2.LINE_AA,
                        )

                else:

                    hotspot_status = "MODEL MISSING"


            # ============================================
            # CALCULATIONS
            # ============================================

            dust_loss = min(
                dust_count * DUST_LOSS_PER,
                DUST_LOSS_MAX
            )

            crack_loss = min(
                crack_count * CRACK_LOSS_PER,
                CRACK_LOSS_MAX
            )

            hotspot_loss = min(
                hotspot_count * HOTSPOT_LOSS_PER,
                HOTSPOT_LOSS_MAX
            )

            total_loss = min(
                dust_loss
                + crack_loss
                + hotspot_loss,
                100.0,
            )

            efficiency = max(
                0.0,
                100.0 - total_loss
            )


            if total_detections == 0:

                health_numeric = 100

                priority = "LOW"

            else:

                # Weighted penalty: hotspots hurt more than dust
                health_numeric = max(
                    0,
                    100 - (
                        dust_count    * DUST_HEALTH_PEN
                        + crack_count   * CRACK_HEALTH_PEN
                        + hotspot_count * HOTSPOT_HEALTH_PEN
                    ),
                )

                if health_numeric < 50:

                    priority = "HIGH"

                else:

                    priority = "MEDIUM"


            health_score = (
                f"{health_numeric}/100"
            )


            avg_confidence = (
                sum(confidence_values)
                / len(confidence_values)
                if confidence_values
                else 0.0
            )


            highest_confidence = (
                max(confidence_values)
                if confidence_values
                else 0.0
            )


            output_rgb = cv2.cvtColor(
                output_bgr,
                cv2.COLOR_BGR2RGB
            )


            # ============================================
            # SAVE STATE
            # ============================================

            st.session_state.analysis_data = {

                "results_img":
                    output_rgb,

                "total_detections":
                    total_detections,

                "dust_count":
                    dust_count,

                "crack_count":
                    crack_count,

                "hotspot_count":
                    hotspot_count,

                "dust_status":
                    dust_status,

                "crack_status":
                    crack_status,

                "hotspot_status":
                    hotspot_status,

                "total_loss":
                    total_loss,

                "efficiency":
                    efficiency,

                "health_numeric":
                    health_numeric,

                "health_score":
                    health_score,

                "priority":
                    priority,

                "avg_confidence":
                    avg_confidence,

                "highest_confidence":
                    highest_confidence,
                "detection_details":
                    detection_details,
                # Store image dimensions so the report can use
                # them without relying on locals() or re-decoding.
                "image_w":
                    image_w,
                "image_h":
                    image_h,
            }

        # ── Rerun AFTER the spinner context exits cleanly ────────
        st.rerun()


    # ========================================================
    # DETAILED REPORT — KEEP IT DIRECTLY UNDER THE IMAGES
    # ========================================================
    with center_col:
        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if st.session_state.analysis_data is not None:

            data = st.session_state.analysis_data


            st.markdown(
                """
                <div class="section-title">
                    📊 Live AI Analysis
                </div>

                <div class="section-subtitle">
                    Results generated from your trained YOLO models.
                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # DETAILED AI INSPECTION REPORT
            # ------------------------------------------------
            details = data.get("detection_details", [])
            dust_loss    = min(data["dust_count"]    * DUST_LOSS_PER,    DUST_LOSS_MAX)
            crack_loss   = min(data["crack_count"]   * CRACK_LOSS_PER,   CRACK_LOSS_MAX)
            hotspot_loss = min(data["hotspot_count"] * HOTSPOT_LOSS_PER, HOTSPOT_LOSS_MAX)

            st.markdown(
                """
                <div style="margin-top:18px;margin-bottom:8px;">
                    <div style="font-size:25px;font-weight:900;color:#F2F7FF;">🔎 Detailed AI Inspection Report</div>
                    <div style="font-size:12px;color:#718198;margin-top:5px;">Model findings, confidence, estimated impact and recommended action for this inspection.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Category-level findings. These values come directly from the model counts
            # and the existing loss calculation in the application.
            category_cards = [
                ("🧹", "Soiling / Dust", data["dust_count"], dust_loss, "#00E5FF", data["dust_status"]),
                ("💥", "Structural Cracks", data["crack_count"], crack_loss, "#FF4757", data["crack_status"]),
                ("🔥", "Thermal Hotspots", data["hotspot_count"], hotspot_loss, "#FFB000", data["hotspot_status"]),
            ]
            c1, c2, c3 = st.columns(3, gap="medium")
            for col, (icon, title, count, loss, accent, status) in zip((c1, c2, c3), category_cards):
                with col:
                    st.markdown(
                        f"""
                        <div style="height:100%;padding:18px;border:1px solid #1A3348;border-radius:16px;
                                    background:linear-gradient(145deg,#081521,#050B12);border-top:3px solid {accent};">
                            <div style="font-size:12px;color:{accent};font-weight:900;letter-spacing:.6px;">{icon} {title.upper()}</div>
                            <div style="font-size:38px;font-weight:900;color:#F4F8FF;margin-top:9px;">{count}</div>
                            <div style="font-size:11px;color:#78879A;">DETECTED REGIONS</div>
                            <div style="margin-top:13px;font-size:13px;color:#D5DFEA;"><b>Status:</b> {status}</div>
                            <div style="margin-top:7px;font-size:13px;color:#D5DFEA;"><b>Estimated impact:</b> {loss:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Region-by-region details are available when the YOLO model returns boxes.
            if details:
                rows = []
                # Use image dimensions stored in session_state (no locals() hack)
                _iw = data.get("image_w") or 1
                _ih = data.get("image_h") or 1
                for idx, item in enumerate(details, 1):
                    cx = ((item["x1"] + item["x2"]) / 2) / _iw * 100
                    cy = ((item["y1"] + item["y2"]) / 2) / _ih * 100
                    rows.append(
                        f"""<tr><td>{idx}</td><td><b>{item["type"]}</b></td><td>{item["confidence"]:.2f}</td>
                        <td>{cx:.0f}% across / {cy:.0f}% down</td><td>{item["area_pct"]:.2f}% of image</td></tr>"""
                    )
                st.markdown(
                    f"""
                    <div style="margin-top:18px;border:1px solid #1A3348;border-radius:16px;overflow:hidden;background:#050B12;">
                        <div style="padding:14px 16px;color:#00E5FF;font-size:14px;font-weight:900;border-bottom:1px solid #173047;">📍 DETECTED REGIONS — REGION-BY-REGION DETAILS</div>
                        <div style="overflow-x:auto;">
                        <table style="width:100%;border-collapse:collapse;font-size:12px;color:#C9D5E2;">
                            <thead><tr style="background:#091724;color:#71869B;text-align:left;">
                                <th style="padding:11px;">#</th><th style="padding:11px;">DEFECT TYPE</th><th style="padding:11px;">CONFIDENCE</th>
                                <th style="padding:11px;">APPROX. LOCATION</th><th style="padding:11px;">BOX AREA</th>
                            </tr></thead>
                            <tbody>{''.join(rows)}</tbody>
                        </table>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Explain what the numbers mean rather than only displaying them.
            interpretation = []
            if data["dust_count"]:
                interpretation.append(f"<b>Soiling:</b> {data["dust_count"]} region(s) detected; the current model calculation assigns {dust_loss:.1f}% estimated loss contribution.")
            if data["crack_count"]:
                interpretation.append(f"<b>Cracks:</b> {data["crack_count"]} region(s) detected; the current model calculation assigns {crack_loss:.1f}% estimated loss contribution.")
            if data["hotspot_count"]:
                interpretation.append(f"<b>Hotspots:</b> {data["hotspot_count"]} region(s) detected; the current model calculation assigns {hotspot_loss:.1f}% estimated loss contribution.")
            if not interpretation:
                interpretation.append("No defect regions were returned above the selected confidence threshold.")

            st.markdown(
                f"""
                <div style="margin-top:18px;display:grid;grid-template-columns:1.15fr .85fr;gap:14px;">
                    <div style="padding:18px;border:1px solid #1A3348;border-radius:16px;background:#07111B;">
                        <div style="color:#00E5FF;font-size:14px;font-weight:900;margin-bottom:10px;">🧠 WHAT THE AI FOUND</div>
                        <div style="color:#B9C6D5;font-size:13px;line-height:1.8;">{'<br>'.join(interpretation)}</div>
                    </div>
                    <div style="padding:18px;border:1px solid #1A3348;border-radius:16px;background:#07111B;">
                        <div style="color:#FFB000;font-size:14px;font-weight:900;margin-bottom:10px;">🛠️ NEXT ACTION</div>
                        <div style="color:#B9C6D5;font-size:13px;line-height:1.8;">
                            {'Prioritize professional inspection of the detected regions, especially cracks or hotspots. Clean soiled areas using an appropriate PV-safe procedure and re-inspect after maintenance.' if (data["crack_count"] or data["hotspot_count"]) else 'Clean the detected soiled regions using a PV-safe procedure, then run another inspection to confirm improvement.'}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if data["total_detections"] == 0:

                st.html(
                    """
                    <div class="report-card good">

                        <h3 style="color:#00FFB0;">
                            ✅ No Defects Detected
                        </h3>

                        <p style="color:#8D9AAC;">
                            No trained defect category was
                            detected above the selected confidence
                            threshold.
                        </p>

                    </div>
                    """
                )

            elif data["priority"] == "HIGH":

                st.html(
                    """
                    <div class="report-card danger">

                        <h3 style="color:#FF4757;">
                            🚨 Critical Defect Detected
                        </h3>

                        <p style="color:#8D9AAC;">
                            Significant defect activity was
                            detected. Maintenance inspection
                            is recommended.
                        </p>

                    </div>
                    """
                )

            else:

                st.html(
                    """
                    <div class="report-card warning">

                        <h3 style="color:#FFB000;">
                            ⚠️ Defect / Soiling Detected
                        </h3>

                        <p style="color:#8D9AAC;">
                            The AI detected one or more
                            potentially problematic regions.
                        </p>

                    </div>
                    """
                )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            st.html(
                f"""
                <div class="metric-grid">

                    <div class="metric-card">

                        <div class="metric-label">
                            Total Defects
                        </div>

                        <div class="metric-value">
                            {data["total_detections"]}
                        </div>

                        <div class="metric-small">
                            LIVE AI RESULT
                        </div>

                    </div>


                    <div class="metric-card">

                        <div class="metric-label">
                            Estimated Loss
                        </div>

                        <div class="metric-value"
                             style="color:#FFB000;">
                            {data["total_loss"]:.1f}%
                        </div>

                        <div class="metric-small">
                            POWER IMPACT
                        </div>

                    </div>


                    <div class="metric-card">

                        <div class="metric-label">
                            Current Efficiency
                        </div>

                        <div class="metric-value"
                             style="color:#00FFB0;">
                            {data["efficiency"]:.1f}%
                        </div>

                        <div class="metric-small">
                            ESTIMATED
                        </div>

                    </div>


                    <div class="metric-card">

                        <div class="metric-label">
                            Avg Confidence
                        </div>

                        <div class="metric-value"
                             style="color:#00E5FF;">
                            {data["avg_confidence"]:.2f}
                        </div>

                        <div class="metric-small">
                            AI CONFIDENCE
                        </div>

                    </div>

                </div>
                """
            )


            # ------------------------------------------------
            # HEALTH / BREAKDOWN / RECOMMENDATION
            # ------------------------------------------------

            health_col, breakdown_col, recommendation_col = (
                st.columns([1, 1.25, 1.25])
            )


            with health_col:

                st.html(
                    f"""
                    <div class="report-card">

                        <div style="
                            text-align:center;
                            color:#AEB8C7;
                            font-weight:800;
                        ">
                            ❤️ PANEL HEALTH SCORE
                        </div>

                        <div class="health-circle">

                            <div class="health-inner">

                                <div class="health-score">
                                    {data["health_numeric"]}
                                </div>

                                <div class="health-label">
                                    / 100
                                </div>

                            </div>

                        </div>

                        <div style="
                            text-align:center;
                            color:#00FFB0;
                            font-weight:800;
                        ">
                            {data["priority"]} PRIORITY
                        </div>

                    </div>
                    """
                )


            with breakdown_col:

                st.html(
                    f"""
                    <div class="report-card">

                        <h3 style="
                            color:#00E5FF;
                            margin-top:0;
                        ">
                            🔍 Detection Breakdown
                        </h3>

                        <p style="color:#B7C1D0;">
                            🧹
                            <b>Soiling / Dust:</b>
                            {data["dust_status"]}
                        </p>

                        <p style="color:#B7C1D0;">
                            💥
                            <b>Cracks:</b>
                            {data["crack_status"]}
                        </p>

                        <p style="color:#B7C1D0;">
                            🔥
                            <b>Hotspots:</b>
                            {data["hotspot_status"]}
                        </p>

                        <hr style="
                            border-color:#1D2B3D;
                        ">

                        <p style="color:#8D9AAC;">
                            Highest Confidence:
                            <b style="color:#00E5FF;">
                                {data["highest_confidence"]:.2f}
                            </b>
                        </p>

                    </div>
                    """
                )


            with recommendation_col:

                if data["priority"] == "HIGH":

                    recommendation = (
                        "Critical inspection is recommended. "
                        "Check detected regions and consider "
                        "professional maintenance."
                    )

                elif data["total_detections"] > 0:

                    recommendation = (
                        "Inspect the detected regions and "
                        "schedule cleaning or maintenance "
                        "as appropriate."
                    )

                else:

                    recommendation = (
                        "No defect was detected. Continue "
                        "routine inspection and cleaning."
                    )


                st.html(
                    f"""
                    <div class="report-card">

                        <h3 style="
                            color:#FFB000;
                            margin-top:0;
                        ">
                            🛠️ AI Recommendation
                        </h3>

                        <p style="
                            color:#AEB8C7;
                            line-height:1.8;
                        ">
                            {recommendation}
                        </p>

                    </div>
                    """
                )


            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            save_col = st.columns(
                [1, 2, 1]
            )[1]

            with save_col:

                if st.button(
                    "💾 SAVE ANALYSIS TO HISTORY",
                    use_container_width=True,
                    key="save_result",
                ):

                    record = {

                        "Timestamp":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "Filename":
                            uploaded_file.name,

                        "Inspection_Mode":
                            inspection_mode,

                        "Dust_Status":
                            data["dust_status"],

                        "Crack_Status":
                            data["crack_status"],

                        "Hotspot_Status":
                            data["hotspot_status"],

                        "Health_Score":
                            data["health_score"],

                        "Priority":
                            data["priority"],

                        "Detections_Count":
                            data["total_detections"],

                        "Estimated_Loss_Pct":
                            f'{data["total_loss"]:.1f}%',
                    }

                    save_record(
                        record,
                        data["results_img"]
                    )


    # ========================================================
    # SOLAR MAINTENANCE & SERVICE GUIDE
    # ========================================================
    st.html("""
    <div style="margin-top:28px;">
        <div class="section-title">🛠️ SOLAR MAINTENANCE GUIDE</div>
        <div class="section-subtitle">Practical steps to keep PV modules clean, safe and productive.</div>
        <div class="metric-grid" style="grid-template-columns:repeat(3,1fr);">
            <div class="report-card">
                <h3 style="color:#00E5FF;margin-top:0;">🧼 Routine Cleaning</h3>
                <p style="color:#AAB7C8;line-height:1.7;font-size:13px;">
                    Clean when soiling becomes visible or when output drops. Prefer
                    early morning/evening, soft brushes or sponges and clean/soft water.
                    Avoid abrasive tools and harsh chemicals.
                </p>
            </div>
            <div class="report-card">
                <h3 style="color:#00FFB0;margin-top:0;">🔎 Visual Inspection</h3>
                <p style="color:#AAB7C8;line-height:1.7;font-size:13px;">
                    Check for cracks, discoloration, loose cables, shading, bird nests,
                    damaged frames and unusual hotspots. Escalate electrical or structural
                    faults to qualified technicians.
                </p>
            </div>
            <div class="report-card warning">
                <h3 style="color:#FFB000;margin-top:0;">⚠️ Safety First</h3>
                <p style="color:#AAB7C8;line-height:1.7;font-size:13px;">
                    Do not open electrical junction boxes or work on live PV wiring.
                    For suspected hotspots, damaged modules or wiring faults, contact
                    a qualified solar service professional.
                </p>
            </div>
        </div>

        <div class="about-card" style="margin-top:14px;">
            <h3 style="color:#00E5FF;margin-top:0;">📍 SERVICE CENTERS / MAINTENANCE SUPPORT</h3>
            <p style="color:#8D9AAC;font-size:12px;line-height:1.6;">
                For the hackathon demo, these are example service categories. Verify
                phone numbers, availability and authorization before publishing them
                as official customer contacts.
            </p>
            <div class="team-grid" style="grid-template-columns:repeat(3,1fr);">
                <div class="team-member" style="text-align:left;">
                    <b style="color:#EAF2FF;">☀️ Paradise Solar Solutions</b><br>
                    <span style="color:#7F8A9D;font-size:11px;">50/37A Pura Padin, Baghambari Rd, Daraganj, Prayagraj 211006</span><br>
                    <span style="color:#00E5FF;font-size:11px;">+91 70072 15437</span>
                </div>
                <div class="team-member" style="text-align:left;">
                    <b style="color:#EAF2FF;">🔧 Om Solar Solutions</b><br>
                    <span style="color:#7F8A9D;font-size:11px;">205H, 3E, 9R, Radha Kunj, Kalindipuram, Prayagraj 211015</span><br>
                    <span style="color:#00E5FF;font-size:11px;">+91 99199 90945</span>
                </div>
                <div class="team-member" style="text-align:left;">
                    <b style="color:#EAF2FF;">🔥 MAHAVEER S SOLAR SERVICE</b><br>
                    <span style="color:#7F8A9D;font-size:11px;">96A/2C/1C, Chak Meera Patti, Harwara, Dhoomanganj, Prayagraj 211011</span><br>
                    <span style="color:#00E5FF;font-size:11px;">+91 84928 62145</span>
                </div>
            </div>
        </div>
    </div>
    """)

# ============================================================
# HISTORY
# ============================================================

with tab_history:

    st.markdown(
        """
        <div class="section-title">
            📁 Inspection History
        </div>

        <div class="section-subtitle">
            Review saved AI inspection reports.
        </div>
        """,
        unsafe_allow_html=True
    )


    if os.path.exists(CSV_FILE):

        try:

            history_df = pd.read_csv(
                CSV_FILE,
                on_bad_lines="skip"
            )

        except Exception:

            history_df = pd.DataFrame()


        if not history_df.empty:

            columns_to_show = [
                "Timestamp",
                "Filename",
                "Inspection_Mode",
                "Dust_Status",
                "Crack_Status",
                "Hotspot_Status",
                "Health_Score",
                "Priority",
                "Detections_Count",
                "Estimated_Loss_Pct",
            ]

            available = [
                c
                for c in columns_to_show
                if c in history_df.columns
            ]

            st.dataframe(
                history_df[available],
                use_container_width=True,
                hide_index=True,
            )


            st.markdown("---")

            st.subheader(
                "🖼️ Saved Visual Reports"
            )


            for _, row in history_df.iterrows():

                title = (
                    f'📌 {row.get("Timestamp", "")} | '
                    f'{row.get("Filename", "")} | '
                    f'Health: {row.get("Health_Score", "N/A")}'
                )

                with st.expander(title):

                    info_col, image_col = st.columns(
                        [1, 1]
                    )

                    with info_col:

                        st.html(
                            f"""
                            <div class="history-item">

                                <p>
                                    <b>Mode:</b>
                                    {row.get("Inspection_Mode", "N/A")}
                                </p>

                                <p>
                                    <b>Dust:</b>
                                    {row.get("Dust_Status", "N/A")}
                                </p>

                                <p>
                                    <b>Cracks:</b>
                                    {row.get("Crack_Status", "N/A")}
                                </p>

                                <p>
                                    <b>Hotspots:</b>
                                    {row.get("Hotspot_Status", "N/A")}
                                </p>

                                <p>
                                    <b>Health:</b>
                                    {row.get("Health_Score", "N/A")}
                                </p>

                                <p>
                                    <b>Priority:</b>
                                    {row.get("Priority", "N/A")}
                                </p>

                                <p>
                                    <b>Power Loss:</b>
                                    {row.get("Estimated_Loss_Pct", "N/A")}
                                </p>

                            </div>
                            """
                        )


                    with image_col:

                        image_path = str(
                            row.get(
                                "Saved_Image_Path",
                                ""
                            )
                        )

                        if (
                            image_path
                            and image_path != "nan"
                            and os.path.exists(image_path)
                        ):

                            st.image(
                                image_path,
                                use_container_width=True
                            )

                        else:

                            st.warning(
                                "Saved image not found."
                            )


            st.markdown("---")

            export_col, clear_col = st.columns(2)

            with export_col:

                csv_data = history_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "📥 EXPORT CSV",
                    data=csv_data,
                    file_name=(
                        "Super_Six_Inspection_Report_"
                        + datetime.now().strftime("%Y%m%d")
                        + ".csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                )


            with clear_col:

                if st.button(
                    "🗑️ CLEAR HISTORY",
                    use_container_width=True,
                    key="clear_history",
                ):

                    try:

                        os.remove(CSV_FILE)

                    except FileNotFoundError:

                        pass

                    initialize_storage()

                    st.rerun()

        else:

            st.info(
                "📁 No saved inspections yet."
            )

    else:

        st.info(
            "📁 No inspection history available."
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

with tab_about:

    st.markdown(
        """
        <div class="section-title">
            ℹ️ About Super Six
        </div>

        <div class="section-subtitle">
            AI-powered solar panel inspection platform.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PROJECT
    # --------------------------------------------------------

    st.html(
        """
        <div class="about-card">

            <h2 style="
                color:white;
                margin-top:0;
            ">
                ☀️ Super Six — Solar Vision AI
            </h2>

            <p style="
                color:#9BA4B5;
                font-size:16px;
                line-height:1.8;
            ">
                Super Six is an AI-powered computer vision
                platform designed to assist solar panel
                inspection and maintenance.
            </p>

            <p style="
                color:#9BA4B5;
                line-height:1.8;
            ">
                Our trained YOLO models analyze inspection
                images to identify surface soiling,
                structural cracks and thermal hotspot patterns.
            </p>

        </div>
        """
    )


    # --------------------------------------------------------
    # TEAM
    # --------------------------------------------------------

    st.html(
        """
        <div class="about-card">

            <h2 style="
                color:white;
                margin-top:0;
            ">
                👥 Team Super Six
            </h2>

            <p style="
                color:#748196;
            ">
                Six minds. One vision. Smarter solar maintenance.
            </p>

            <div class="team-grid">

                <div class="team-member">
                    <div style="font-size:28px;">
                        👨‍💻
                    </div>
                    <b>Abhijeet</b>
                </div>

                <div class="team-member">
                    <div style="font-size:28px;">
                        👩‍💻
                    </div>
                    <b>Nidhi</b>
                </div>

                <div class="team-member">
                    <div style="font-size:28px;">
                        👨‍💻
                    </div>
                    <b>Ansh</b>
                </div>

                <div class="team-member">
                    <div style="font-size:28px;">
                        👨‍💻
                    </div>
                    <b>Anubhav</b>
                </div>

                <div class="team-member">
                    <div style="font-size:28px;">
                        👩‍💻
                    </div>
                    <b>Akanksha</b>
                </div>

                <div class="team-member">
                    <div style="font-size:28px;">
                        👩‍💻
                    </div>
                    <b>Trisha</b>
                </div>

            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    tech1, tech2 = st.columns(2)

    with tech1:

        st.html(
            """
            <div class="about-card">

                <h3 style="color:#00E5FF;">
                    🧠 AI Technology
                </h3>

                <p style="color:#AEB8C7;">
                    🤖 YOLO Object Detection
                </p>

                <p style="color:#AEB8C7;">
                    👁️ Computer Vision
                </p>

                <p style="color:#AEB8C7;">
                    🔬 OpenCV
                </p>

                <p style="color:#AEB8C7;">
                    🎯 Custom Trained Models
                </p>

            </div>
            """
        )


    with tech2:

        st.html(
            """
            <div class="about-card">

                <h3 style="color:#FFB000;">
                    ⚙️ Application Stack
                </h3>

                <p style="color:#AEB8C7;">
                    🐍 Python
                </p>

                <p style="color:#AEB8C7;">
                    🌐 Streamlit
                </p>

                <p style="color:#AEB8C7;">
                    📊 Pandas
                </p>

                <p style="color:#AEB8C7;">
                    🖼️ PIL
                </p>

            </div>
            """
        )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.html(
        """
        <div class="about-card">

            <h2 style="
                color:white;
                margin-top:0;
            ">
                🚀 Key Features
            </h2>

            <div class="team-grid">

                <div class="team-member">
                    🧹<br>
                    <b>Soiling Detection</b>
                </div>

                <div class="team-member">
                    💥<br>
                    <b>Crack Detection</b>
                </div>

                <div class="team-member">
                    🔥<br>
                    <b>Hotspot Detection</b>
                </div>

                <div class="team-member">
                    📊<br>
                    <b>Health Scoring</b>
                </div>

                <div class="team-member">
                    ⚡<br>
                    <b>Power Loss Estimation</b>
                </div>

                <div class="team-member">
                    📁<br>
                    <b>Inspection History</b>
                </div>

            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # LEAD DEVELOPER
    # --------------------------------------------------------

    st.html(
        """
        <div class="lead-card">

            <div style="
                font-size:35px;
                margin-bottom:8px;
            ">
                👑
            </div>

            <div style="
                color:#FFB000;
                font-size:12px;
                font-weight:800;
                letter-spacing:1.5px;
            ">
                LEAD DEVELOPER
            </div>

            <div style="
                color:white;
                font-size:25px;
                font-weight:900;
                margin-top:6px;
            ">
                Abhijeet Singh
            </div>

            <div style="
                color:#78869A;
                font-size:12px;
                margin-top:5px;
            ">
                Super Six • Solar Vision AI
            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        ☀️ <b>SUPER SIX — SOLAR VISION AI</b>

        <br><br>

        Detect • Analyze • Predict • Maintain

        <br><br>

        Built for the future of renewable energy ⚡

    </div>
    """
)