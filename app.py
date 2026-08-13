import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import os
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Vision AI | Super Six",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

:root {
    --bg: #03050a;
    --card: #0b1220;
    --border: rgba(255,255,255,0.10);

    --cyan: #00e5ff;
    --blue: #4f8cff;
    --green: #00ff9d;
    --yellow: #ffd43b;
    --orange: #ff9f1c;
    --red: #ff3864;
    --purple: #a855f7;
}


/* =========================================================
   MAIN APP
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(0,229,255,0.09),
            transparent 25%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(168,85,247,0.09),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 90%,
            rgba(0,255,157,0.05),
            transparent 30%
        ),
        #03050a;

    color: #f8fafc;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background-attachment: fixed;
}


/* =========================================================
   ANIMATED GRID
   ========================================================= */

.stApp::before {
    content: "";

    position: fixed;
    inset: 0;

    background-image:
        linear-gradient(
            rgba(0,229,255,0.035) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(0,229,255,0.035) 1px,
            transparent 1px
        );

    background-size: 45px 45px;

    pointer-events: none;

    animation: gridMove 18s linear infinite;

    z-index: 0;
}

@keyframes gridMove {

    from {
        transform: translateY(0);
    }

    to {
        transform: translateY(45px);
    }
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #050912 0%,
            #02040a 100%
        ) !important;

    border-right:
        1px solid rgba(0,229,255,0.15);

    box-shadow:
        10px 0 40px rgba(0,0,0,0.35);
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1 {

    font-size: 3rem !important;

    font-weight: 850 !important;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #00e5ff,
            #00ff9d
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    letter-spacing: -0.04em;

    text-shadow:
        0 0 30px rgba(0,229,255,0.18);
}

h2 {
    font-weight: 750 !important;
}

h3 {
    font-weight: 700 !important;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    position: relative;

    padding: 38px;

    margin-bottom: 25px;

    border-radius: 24px;

    overflow: hidden;

    background:
        radial-gradient(
            circle at 88% 50%,
            rgba(255,193,7,0.20),
            transparent 24%
        ),
        linear-gradient(
            135deg,
            rgba(15,25,42,0.96),
            rgba(3,7,14,0.94)
        );

    border:
        1px solid rgba(0,229,255,0.18);

    box-shadow:
        0 20px 70px rgba(0,0,0,0.45);
}

.hero::before {

    content: "";

    position: absolute;

    width: 250px;
    height: 250px;

    right: 7%;
    top: -80px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            #ffd43b 0%,
            #ff9f1c 35%,
            rgba(255,159,28,0.08) 65%,
            transparent 70%
        );

    animation:
        solarPulse 4s ease-in-out infinite;
}

@keyframes solarPulse {

    0%,100% {
        transform: scale(0.95);
        opacity: 0.7;
    }

    50% {
        transform: scale(1.08);
        opacity: 1;
    }
}

.hero-title {

    position: relative;

    font-size: 42px;

    font-weight: 850;

    letter-spacing: -1.5px;

    margin-top: 15px;

    color: white;

    z-index: 2;
}

.hero-subtitle {

    position: relative;

    color: #94a3b8;

    font-size: 16px;

    margin-top: 8px;

    z-index: 2;
}


/* =========================================================
   STATUS
   ========================================================= */

.status-online {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 7px 14px;

    border-radius: 999px;

    background:
        rgba(0,255,157,0.08);

    border:
        1px solid rgba(0,255,157,0.35);

    color: #00ff9d;

    font-size: 12px;

    font-weight: 800;

    letter-spacing: 0.05em;
}

.status-dot {

    width: 8px;
    height: 8px;

    background: #00ff9d;

    border-radius: 50%;

    box-shadow:
        0 0 8px #00ff9d;

    animation:
        pulse 1.5s infinite;
}

@keyframes pulse {

    0% {
        transform: scale(0.8);
        opacity: 0.6;
    }

    50% {
        transform: scale(1.2);
        opacity: 1;
    }

    100% {
        transform: scale(0.8);
        opacity: 0.6;
    }
}


/* =========================================================
   CARDS
   ========================================================= */

.metric-container,
.ai-card,
.about-card,
.report-card,
.report-card-warning {

    background:
        linear-gradient(
            145deg,
            rgba(20,30,48,0.86),
            rgba(5,10,18,0.78)
        );

    border:
        1px solid rgba(255,255,255,0.09);

    border-radius: 18px;

    padding: 22px;

    backdrop-filter: blur(18px);

    box-shadow:
        0 10px 40px rgba(0,0,0,0.35),
        inset 0 1px rgba(255,255,255,0.05);

    transition:
        transform 0.25s ease,
        border 0.25s ease,
        box-shadow 0.25s ease;
}

.metric-container:hover,
.ai-card:hover,
.about-card:hover {

    transform: translateY(-4px);

    border-color:
        rgba(0,229,255,0.42);

    box-shadow:
        0 15px 50px rgba(0,229,255,0.12);
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {

    border-radius: 12px !important;

    border:
        1px solid rgba(0,229,255,0.45) !important;

    background:
        linear-gradient(
            135deg,
            #00e5ff,
            #4f8cff
        ) !important;

    color: #001018 !important;

    font-weight: 800 !important;

    padding: 10px 22px !important;

    box-shadow:
        0 0 20px rgba(0,229,255,0.15);

    transition:
        all 0.25s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 0 30px rgba(0,229,255,0.35);

    border-color:
        #00e5ff !important;
}


/* =========================================================
   TABS
   ========================================================= */

.stTabs [data-baseweb="tab-list"] {

    gap: 5px;

    background:
        rgba(5,10,18,0.75);

    border:
        1px solid rgba(255,255,255,0.06);

    border-radius: 14px;

    padding: 5px;
}

.stTabs [data-baseweb="tab"] {

    border-radius: 10px;

    color: #94a3b8;

    padding: 12px 20px;

    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {

    color: white !important;

    background:
        linear-gradient(
            135deg,
            rgba(0,229,255,0.15),
            rgba(79,140,255,0.12)
        );
}


/* =========================================================
   IMAGE
   ========================================================= */

[data-testid="stImage"] {

    border-radius: 16px;

    overflow: hidden;

    border:
        1px solid rgba(0,229,255,0.15);

    box-shadow:
        0 10px 45px rgba(0,0,0,0.4);
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] section {

    background:
        linear-gradient(
            135deg,
            rgba(0,229,255,0.04),
            rgba(79,140,255,0.04)
        );

    border:
        1px dashed rgba(0,229,255,0.35);

    border-radius: 14px;

    transition: 0.3s ease;
}

[data-testid="stFileUploader"] section:hover {

    border-color:
        #00e5ff;

    box-shadow:
        0 0 25px rgba(0,229,255,0.10);
}


/* =========================================================
   MODEL STATUS
   ========================================================= */

.model-pill {

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 11px 14px;

    margin: 7px 0;

    border-radius: 10px;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.06);

    color: #e2e8f0;
}

.model-online {

    color: #00ff9d;

    font-size: 11px;

    font-weight: 800;
}


/* =========================================================
   SCANNER
   ========================================================= */

.scan-box {

    position: relative;

    overflow: hidden;

    border-radius: 16px;

    border:
        1px solid rgba(0,229,255,0.35);

    background:
        #02060d;
}

.scan-box::after {

    content: "";

    position: absolute;

    left: 0;
    right: 0;

    height: 3px;

    background:
        linear-gradient(
            90deg,
            transparent,
            #00e5ff,
            #00ff9d,
            transparent
        );

    box-shadow:
        0 0 20px #00e5ff;

    animation:
        scanning 2.2s linear infinite;
}

@keyframes scanning {

    0% {
        top: 0;
        opacity: 0;
    }

    10% {
        opacity: 1;
    }

    90% {
        opacity: 1;
    }

    100% {
        top: 100%;
        opacity: 0;
    }
}


/* =========================================================
   EXPANDERS
   ========================================================= */

[data-testid="stExpander"] {

    background:
        rgba(10,16,27,0.75);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 14px;

    overflow: hidden;
}


/* =========================================================
   DIVIDER
   ========================================================= */

hr {

    border: none !important;

    height: 1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(0,229,255,0.35),
            transparent
        );
}


/* =========================================================
   SCROLLBAR
   ========================================================= */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #02040a;
}

::-webkit-scrollbar-thumb {

    background:
        linear-gradient(
            #00e5ff,
            #4f8cff
        );

    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

CSV_FILE = "inspection_history.csv"
IMAGE_DIR = "saved_images"

EXPECTED_COLUMNS = [
    "Timestamp",
    "Filename",
    "Inspection_Mode",
    "Panel_Status",
    "Dust_Status",
    "Crack_Status",
    "Hotspot_Status",
    "Health_Score",
    "Priority",
    "Detections_Count",
    "Estimated_Loss_Pct",
    "Saved_Image_Path"
]


# ============================================================
# DATA STORAGE
# ============================================================

def init_data_storage():

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    if os.path.exists(CSV_FILE):

        try:

            existing_df = pd.read_csv(
                CSV_FILE,
                nrows=1
            )

            existing_columns = list(
                existing_df.columns
            )

            if existing_columns != EXPECTED_COLUMNS:

                backup_name = (
                    f"inspection_history_old_"
                    f"{int(datetime.now().timestamp())}.csv"
                )

                os.rename(
                    CSV_FILE,
                    backup_name
                )

                pd.DataFrame(
                    columns=EXPECTED_COLUMNS
                ).to_csv(
                    CSV_FILE,
                    index=False
                )

        except Exception:

            pd.DataFrame(
                columns=EXPECTED_COLUMNS
            ).to_csv(
                CSV_FILE,
                index=False
            )

    else:

        pd.DataFrame(
            columns=EXPECTED_COLUMNS
        ).to_csv(
            CSV_FILE,
            index=False
        )


def save_inspection_record(
    record_data,
    image_array
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    safe_filename = (
        record_data["Filename"]
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    image_filename = (
        f"{timestamp}_{safe_filename}"
    )

    image_path = os.path.join(
        IMAGE_DIR,
        image_filename
    )

    Image.fromarray(
        image_array
    ).save(image_path)

    record_data["Saved_Image_Path"] = image_path

    df = pd.DataFrame(
        [record_data],
        columns=EXPECTED_COLUMNS
    )

    df.to_csv(
        CSV_FILE,
        mode="a",
        header=False,
        index=False
    )

    st.toast(
        "Analysis saved successfully!",
        icon="✅"
    )


init_data_storage()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_all_models():

    loaded = {}

    model_files = {
        "panel": "panel_detector.pt",
        "dust": "dust_model.pt",
        "crack": "crack_model.pt",
        "hotspot": "hotspot_model.pt"
    }

    for key, filename in model_files.items():

        if os.path.exists(filename):

            try:

                loaded[key] = YOLO(filename)

            except Exception as error:

                loaded[key] = None

                print(
                    f"Error loading {filename}: {error}"
                )

        else:

            loaded[key] = None

    return loaded


models = load_all_models()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_solar_panel(
    image,
    loaded_models
):

    if loaded_models.get("panel") is not None:

        try:

            results = loaded_models["panel"](
                image,
                conf=0.30,
                verbose=False
            )

            return (
                len(results[0].boxes) > 0
            )

        except Exception:
            pass

    # Fallback OpenCV validation

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if sharpness < 10:
        return False

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=15
    )

    return (
        lines is not None
        and len(lines) >= 2
    )


def run_model(
    model,
    image,
    confidence
):

    if model is None:
        return None

    try:

        result = model(
            image,
            conf=confidence,
            verbose=False
        )[0]

        return result

    except Exception as error:

        st.error(
            f"AI model error: {error}"
        )

        return None


def draw_detections(
    image,
    result,
    label,
    color
):

    output = image.copy()

    confidences = []

    if result is None:
        return output, confidences

    if result.boxes is None:
        return output, confidences

    for box in result.boxes:

        coordinates = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = map(
            int,
            coordinates
        )

        confidence = float(
            box.conf[0]
            .cpu()
            .numpy()
        )

        confidences.append(
            confidence
        )

        # Bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # Label
        text = (
            f"{label} "
            f"{confidence:.2f}"
        )

        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )

        text_y = max(
            y1 - 8,
            th + 8
        )

        cv2.rectangle(
            output,
            (
                x1,
                text_y - th - 8
            ),
            (
                x1 + tw + 10,
                text_y + 2
            ),
            color,
            -1
        )

        cv2.putText(
            output,
            text,
            (
                x1 + 5,
                text_y - 3
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return output, confidences


def get_health_score(
    detections
):

    if detections == 0:
        return 100

    return max(
        0,
        100 - detections * 15
    )


def get_health_color(
    score
):

    if score >= 80:
        return "#00ff9d"

    if score >= 50:
        return "#ffd43b"

    return "#ff3864"


def get_priority(
    score
):

    if score >= 80:
        return "LOW"

    if score >= 50:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Logo
    if os.path.exists("logo.png"):

        st.image(
            "logo.png",
            width=180
        )

    elif os.path.exists("logo.jpg"):

        st.image(
            "logo.jpg",
            width=180
        )

    else:

        st.html(
            """
            <div style="
                text-align:center;
                font-size:60px;
                margin:10px;
            ">
                ☀️
            </div>
            """
        )


    st.html(
        """
        <div style="
            text-align:center;
            font-size:23px;
            font-weight:850;
            letter-spacing:2px;
            color:white;
        ">
            SOLAR VISION
        </div>

        <div style="
            text-align:center;
            color:#64748b;
            font-size:10px;
            letter-spacing:3px;
            margin-top:4px;
        ">
            SUPER SIX • AI INSPECTION
        </div>
        """
    )


    st.markdown("---")


    # System status
    st.html(
        """
        <div class="status-online">
            <span class="status-dot"></span>
            SYSTEM ONLINE
        </div>
        """
    )


    st.markdown("### ⚙️ Inspection Control")


    inspection_mode = st.selectbox(
        "Inspection Mode",
        [
            "Dust Detection",
            "Crack Detection",
            "Hotspot Detection",
            "Comprehensive Analysis"
        ]
    )


    if inspection_mode == "Hotspot Detection":

        st.warning(
            """
            ⚠️ Thermal Image Required

            Use a thermal / infrared image
            for accurate hotspot detection.
            """
        )


    elif inspection_mode == "Comprehensive Analysis":

        st.info(
            """
            💡 Comprehensive Mode

            Checks dust, cracks and hotspots.
            Thermal imagery is recommended
            for hotspot detection.
            """
        )


    uploaded_file = st.file_uploader(
        "📤 Upload Inspection Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    confidence_threshold = st.slider(
        "🎯 AI Confidence",
        min_value=0.05,
        max_value=1.00,
        value=0.40,
        step=0.05
    )


    st.markdown("---")

    st.markdown("### 🧠 AI Model Status")


    model_names = {
        "panel": "☀️ Panel Detector",
        "dust": "🧹 Dust Detector",
        "crack": "🔬 Crack Detector",
        "hotspot": "🔥 Hotspot Detector"
    }


    for key, name in model_names.items():

        if models.get(key) is not None:

            status = (
                '<span class="model-online">'
                '● ONLINE'
                '</span>'
            )

        else:

            status = (
                '<span style="'
                'color:#ff3864;'
                'font-size:11px;'
                'font-weight:800;">'
                '● OFFLINE'
                '</span>'
            )

        st.html(
            f"""
            <div class="model-pill">

                <span>
                    {name}
                </span>

                {status}

            </div>
            """
        )


    st.markdown("---")


    st.html(
        """
        <div style="
            text-align:center;
            color:#64748b;
            font-size:11px;
        ">
            SUPER SIX • Solar AI Intelligence
            <br><br>
            Lead Developer: Abhijeet Singh
        </div>
        """
    )


# ============================================================
# TABS
# ============================================================

tab_dashboard, tab_about, tab_history = st.tabs(
    [
        "📊 AI Dashboard",
        "🚀 About Project",
        "📁 Inspection History"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tab_dashboard:

    st.html(
        """
        <div class="hero">

            <div class="status-online">
                <span class="status-dot"></span>
                AI INSPECTION ENGINE ONLINE
            </div>

            <div class="hero-title">
                ☀️ Solar Vision
            </div>

            <div class="hero-subtitle">
                Intelligent photovoltaic inspection
                & predictive maintenance platform
            </div>

        </div>
        """
    )


    if "analyzed_data" not in st.session_state:

        st.session_state.analyzed_data = None


    if "last_filename" not in st.session_state:

        st.session_state.last_filename = None


    # ========================================================
    # UPLOAD
    # ========================================================

    if uploaded_file is None:

        st.html(
            """
            <div class="ai-card"
                 style="
                    text-align:center;
                    padding:65px 30px;
                 ">

                <div style="
                    font-size:65px;
                ">
                    ☀️
                </div>

                <div style="
                    font-size:28px;
                    font-weight:850;
                    color:white;
                    margin-top:15px;
                ">
                    Ready for Inspection
                </div>

                <div style="
                    color:#64748b;
                    margin-top:10px;
                    font-size:14px;
                ">
                    Upload a solar panel image
                    from the sidebar to start
                    AI-powered inspection.
                </div>

                <div style="
                    margin-top:25px;
                ">

                    <span class="status-online">
                        🧹 DUST
                    </span>

                    <span class="status-online">
                        🔬 CRACKS
                    </span>

                    <span class="status-online">
                        🔥 HOTSPOTS
                    </span>

                </div>

            </div>
            """
        )

    else:

        # Reset analysis if new image
        if (
            st.session_state.last_filename
            != uploaded_file.name
        ):

            st.session_state.last_filename = (
                uploaded_file.name
            )

            st.session_state.analyzed_data = None


        file_bytes = np.asarray(
            bytearray(
                uploaded_file.getvalue()
            ),
            dtype=np.uint8
        )


        original_img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if original_img is None:

            st.error(
                "Unable to read the uploaded image."
            )

            st.stop()


        rgb_img = cv2.cvtColor(
            original_img,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # IMAGE PREVIEW
        # ====================================================

        col1, col2 = st.columns(
            2,
            gap="large"
        )


        with col1:

            st.html(
                """
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    margin-bottom:10px;
                ">

                    <h3 style="
                        margin:0;
                        color:white;
                    ">
                        📸 Input Frame
                    </h3>

                    <span class="status-online">
                        ORIGINAL
                    </span>

                </div>
                """
            )

            st.image(
                rgb_img,
                use_container_width=True
            )


        with col2:

            if st.session_state.analyzed_data is not None:

                st.html(
                    """
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        margin-bottom:10px;
                    ">

                        <h3 style="
                            margin:0;
                            color:white;
                        ">
                            🛰️ AI Detection
                        </h3>

                        <span class="status-online">
                            ANALYZED
                        </span>

                    </div>
                    """
                )

                st.image(
                    st.session_state.analyzed_data[
                        "results_img"
                    ],
                    use_container_width=True
                )

            else:

                st.html(
                    """
                    <div style="
                        height:300px;

                        display:flex;
                        align-items:center;
                        justify-content:center;

                        text-align:center;

                        border:
                            1px dashed
                            rgba(0,229,255,.25);

                        border-radius:16px;

                        color:#64748b;
                    ">

                        <div>

                            <div style="
                                font-size:50px;
                            ">
                                🛰️
                            </div>

                            <div style="
                                font-size:17px;
                                font-weight:700;
                                color:#94a3b8;
                            ">
                                AI Detection Output
                            </div>

                            <div style="
                                font-size:12px;
                                margin-top:6px;
                            ">
                                Run analysis to visualize
                                detections
                            </div>

                        </div>

                    </div>
                    """
                )


        # ====================================================
        # ANALYSIS BUTTON
        # ====================================================

        st.markdown("")


        run_analysis = st.button(
            "🚀 RUN AI ANALYSIS",
            key="run_ai_btn"
        )


        if run_analysis:

            st.session_state.analyzed_data = None


            scanner = st.empty()


            scanner.html(
                """
                <div class="scan-box">

                    <div style="
                        padding:42px;
                        text-align:center;
                    ">

                        <div style="
                            font-size:42px;
                        ">
                            🛰️
                        </div>

                        <div style="
                            color:#00e5ff;
                            font-size:20px;
                            font-weight:800;
                            margin-top:8px;
                        ">
                            AI VISION SCANNING
                        </div>

                        <div style="
                            color:#64748b;
                            font-size:13px;
                            margin-top:7px;
                        ">
                            Detecting panel boundaries
                            • dust • cracks • hotspots
                        </div>

                    </div>

                </div>
                """
            )


            with st.spinner(
                "Running YOLO vision models..."
            ):

                # --------------------------------------------
                # PANEL
                # --------------------------------------------

                is_panel_present = (
                    validate_solar_panel(
                        original_img,
                        models
                    )
                )


                # --------------------------------------------
                # INITIAL VALUES
                # --------------------------------------------

                results_img = rgb_img.copy()

                total_detections = 0

                confidences = []

                num_dust = 0
                num_cracks = 0
                num_hotspots = 0

                dust_status = "UNTESTED"
                crack_status = "UNTESTED"
                hotspot_status = "UNTESTED"


                # ---> FIX: ONLY RUN YOLO IF A PANEL IS DETECTED! <---
                if is_panel_present:

                    # --------------------------------------------
                    # DUST
                    # --------------------------------------------

                    if inspection_mode in [
                        "Dust Detection",
                        "Comprehensive Analysis"
                    ]:

                        if models["dust"] is not None:

                            result = run_model(
                                models["dust"],
                                original_img,
                                confidence_threshold
                            )

                            if result is not None:

                                num_dust = len(
                                    result.boxes
                                )

                                total_detections += (
                                    num_dust
                                )

                                if num_dust > 0:

                                    dust_status = (
                                        f"DETECTED "
                                        f"({num_dust} regions)"
                                    )

                                    results_img, confs = (
                                        draw_detections(
                                            results_img,
                                            result,
                                            "DUST",
                                            (0, 229, 255)
                                        )
                                    )

                                    confidences.extend(
                                        confs
                                    )

                                else:

                                    dust_status = "NONE"

                        else:

                            dust_status = "MODEL MISSING"


                    # --------------------------------------------
                    # CRACK
                    # --------------------------------------------

                    if inspection_mode in [
                        "Crack Detection",
                        "Comprehensive Analysis"
                    ]:

                        if models["crack"] is not None:

                            result = run_model(
                                models["crack"],
                                original_img,
                                confidence_threshold
                            )

                            if result is not None:

                                num_cracks = len(
                                    result.boxes
                                )

                                total_detections += (
                                    num_cracks
                                )

                                if num_cracks > 0:

                                    crack_status = (
                                        f"DETECTED "
                                        f"({num_cracks} regions)"
                                    )

                                    results_img, confs = (
                                        draw_detections(
                                            results_img,
                                            result,
                                            "CRACK",
                                            (255, 70, 100)
                                        )
                                    )

                                    confidences.extend(
                                        confs
                                    )

                                else:

                                    crack_status = "NONE"

                        else:

                            crack_status = "MODEL MISSING"


                    # --------------------------------------------
                    # HOTSPOT
                    # --------------------------------------------

                    if inspection_mode in [
                        "Hotspot Detection",
                        "Comprehensive Analysis"
                    ]:

                        if models["hotspot"] is not None:

                            result = run_model(
                                models["hotspot"],
                                original_img,
                                confidence_threshold
                            )

                            if result is not None:

                                num_hotspots = len(
                                    result.boxes
                                )

                                total_detections += (
                                    num_hotspots
                                )

                                if num_hotspots > 0:

                                    hotspot_status = (
                                        f"DETECTED "
                                        f"({num_hotspots} regions)"
                                    )

                                    results_img, confs = (
                                        draw_detections(
                                            results_img,
                                            result,
                                            "HOTSPOT",
                                            (255, 120, 0)
                                        )
                                    )

                                    confidences.extend(
                                        confs
                                    )

                                else:

                                    hotspot_status = "NONE"

                        else:

                            hotspot_status = "MODEL MISSING"


                # =================================================
                # LOSS ESTIMATION
                # =================================================

                dust_loss = min(
                    num_dust * 2.5,
                    30.0
                )

                crack_loss = min(
                    num_cracks * 5.0,
                    40.0
                )

                hotspot_loss = min(
                    num_hotspots * 15.0,
                    60.0
                )

                total_loss = min(
                    dust_loss
                    + crack_loss
                    + hotspot_loss,
                    100.0
                )

                efficiency = (
                    100.0 - total_loss
                )


                # =================================================
                # HEALTH
                # =================================================

                if not is_panel_present:

                    health_score = "N/A"
                    priority = "INVALID"
                    panel_status = "NOT DETECTED"

                else:

                    score = get_health_score(
                        total_detections
                    )

                    health_score = (
                        f"{score}/100"
                    )

                    priority = get_priority(
                        score
                    )

                    if total_detections == 0:

                        panel_status = (
                            "DETECTED (CLEAN)"
                        )

                    else:

                        panel_status = (
                            "DETECTED (DEFECTIVE)"
                        )


                # =================================================
                # STORE RESULT
                # =================================================

                st.session_state.analyzed_data = {

                    "results_img":
                        results_img,

                    "is_panel_present":
                        is_panel_present,

                    "total_detections":
                        total_detections,

                    "confidences":
                        confidences,

                    "num_dust":
                        num_dust,

                    "num_cracks":
                        num_cracks,

                    "num_hotspots":
                        num_hotspots,

                    "dust_status":
                        dust_status,

                    "crack_status":
                        crack_status,

                    "hotspot_status":
                        hotspot_status,

                    "dust_loss":
                        dust_loss,

                    "crack_loss":
                        crack_loss,

                    "hotspot_loss":
                        hotspot_loss,

                    "total_estimated_loss":
                        total_loss,

                    "current_efficiency":
                        efficiency,

                    "health_score":
                        health_score,

                    "priority":
                        priority,

                    "panel_status":
                        panel_status
                }


            scanner.empty()

            st.rerun()


        # ====================================================
        # RESULTS
        # ====================================================

        if st.session_state.analyzed_data is not None:

            data = (
                st.session_state.analyzed_data
            )


            st.markdown("---")


            # =================================================
            # INVALID PANEL
            # =================================================

            if not data["is_panel_present"]:

                st.error(
                    "❌ No Solar Panel Detected"
                )

                st.html(
                    """
                    <div class="ai-card">

                        <h3 style="
                            color:#ff3864;
                        ">
                            ⚠️ Image Validation Failed
                        </h3>

                        <p style="
                            color:#94a3b8;
                        ">
                            The AI system could not identify
                            a valid solar panel in this image.
                        </p>

                        <span class="status-online"
                              style="
                                color:#ff3864;
                                border-color:
                                rgba(255,56,100,.35);
                              ">
                            PANEL NOT DETECTED
                        </span>

                    </div>
                    """
                )


            else:

                # =================================================
                # STATUS
                # =================================================

                if data["total_detections"] == 0:

                    st.success(
                        "✅ AI Analysis Complete — Panel appears healthy."
                    )

                else:

                    st.warning(
                        "⚠️ Potential defects or soiling detected."
                    )


                # =================================================
                # METRICS
                # =================================================

                st.markdown(
                    "## 📊 Inspection Intelligence"
                )


                m1, m2, m3, m4 = st.columns(4)


                if data["confidences"]:

                    avg_conf = (
                        sum(
                            data["confidences"]
                        )
                        /
                        len(
                            data["confidences"]
                        )
                    )

                    avg_conf_text = (
                        f"{avg_conf:.2f}"
                    )

                else:

                    avg_conf_text = "N/A"


                with m1:

                    st.html(
                        f"""
                        <div class="metric-container">

                            <small style="
                                color:#94a3b8;
                            ">
                                🔎 TOTAL DEFECTS
                            </small>

                            <h2 style="
                                font-size:34px;
                                color:white;
                                margin:8px 0;
                            ">
                                {data['total_detections']}
                            </h2>

                            <small style="
                                color:#64748b;
                            ">
                                AI detected regions
                            </small>

                        </div>
                        """
                    )


                with m2:

                    loss = data[
                        "total_estimated_loss"
                    ]

                    if loss > 20:

                        loss_color = "#ff3864"

                    elif loss > 5:

                        loss_color = "#ffd43b"

                    else:

                        loss_color = "#00ff9d"


                    st.html(
                        f"""
                        <div class="metric-container">

                            <small style="
                                color:#94a3b8;
                            ">
                                ⚡ ESTIMATED LOSS
                            </small>

                            <h2 style="
                                font-size:34px;
                                color:{loss_color};
                                margin:8px 0;
                            ">
                                {loss:.1f}%
                            </h2>

                            <small style="
                                color:#64748b;
                            ">
                                Estimated energy impact
                            </small>

                        </div>
                        """
                    )


                with m3:

                    st.html(
                        f"""
                        <div class="metric-container">

                            <small style="
                                color:#94a3b8;
                            ">
                                ☀️ EFFICIENCY
                            </small>

                            <h2 style="
                                font-size:34px;
                                color:#00ff9d;
                                margin:8px 0;
                            ">
                                {data['current_efficiency']:.1f}%
                            </h2>

                            <small style="
                                color:#64748b;
                            ">
                                Estimated current output
                            </small>

                        </div>
                        """
                    )


                with m4:

                    st.html(
                        f"""
                        <div class="metric-container">

                            <small style="
                                color:#94a3b8;
                            ">
                                🎯 AI CONFIDENCE
                            </small>

                            <h2 style="
                                font-size:34px;
                                color:#00e5ff;
                                margin:8px 0;
                            ">
                                {avg_conf_text}
                            </h2>

                            <small style="
                                color:#64748b;
                            ">
                                Detection certainty
                            </small>

                        </div>
                        """
                    )


                # =================================================
                # HEALTH GAUGE
                # =================================================

                score = int(
                    data["health_score"]
                    .split("/")[0]
                )

                health_color = get_health_color(
                    score
                )


                if score >= 80:

                    health_label = "EXCELLENT"

                elif score >= 50:

                    health_label = "ATTENTION"

                else:

                    health_label = "CRITICAL"


                st.markdown("")


                st.html(
                    f"""
                    <div class="ai-card">

                        <div style="
                            display:flex;
                            justify-content:space-between;
                            align-items:center;
                            flex-wrap:wrap;
                            gap:30px;
                        ">

                            <div>

                                <div style="
                                    color:#94a3b8;
                                    font-size:13px;
                                    font-weight:800;
                                    letter-spacing:1px;
                                ">
                                    SOLAR PANEL HEALTH INDEX
                                </div>

                                <div style="
                                    font-size:46px;
                                    font-weight:900;
                                    color:{health_color};
                                    margin-top:6px;
                                ">
                                    {score}
                                    <span style="
                                        font-size:20px;
                                        color:#64748b;
                                    ">
                                        /100
                                    </span>
                                </div>

                                <div style="
                                    color:{health_color};
                                    font-weight:850;
                                    letter-spacing:1px;
                                ">
                                    ● {health_label}
                                </div>

                                <div style="
                                    color:#64748b;
                                    margin-top:10px;
                                    font-size:13px;
                                ">
                                    Panel status:
                                    {data['panel_status']}
                                </div>

                            </div>


                            <div style="
                                width:145px;
                                height:145px;
                                border-radius:50%;

                                background:
                                    conic-gradient(
                                        {health_color}
                                        {score}%,
                                        #172033
                                        {score}%
                                    );

                                display:flex;
                                align-items:center;
                                justify-content:center;

                                box-shadow:
                                    0 0 40px
                                    {health_color}33;
                            ">

                                <div style="
                                    width:115px;
                                    height:115px;

                                    border-radius:50%;

                                    background:#050a12;

                                    display:flex;
                                    align-items:center;
                                    justify-content:center;

                                    color:white;

                                    font-size:26px;
                                    font-weight:850;
                                ">
                                    {score}%
                                </div>

                            </div>

                        </div>

                    </div>
                    """
                )


                # =================================================
                # DETECTION BREAKDOWN
                # =================================================

                st.markdown(
                    "### 🔬 AI Detection Breakdown"
                )


                d1, d2, d3 = st.columns(3)


                with d1:

                    st.html(
                        f"""
                        <div class="ai-card">

                            <div style="
                                font-size:32px;
                            ">
                                🧹
                            </div>

                            <div style="
                                color:#94a3b8;
                                font-size:12px;
                                font-weight:700;
                                margin-top:8px;
                            ">
                                DUST / SOILING
                            </div>

                            <div style="
                                font-size:20px;
                                font-weight:800;
                                margin-top:7px;
                                color:#00e5ff;
                            ">
                                {data['dust_status']}
                            </div>

                        </div>
                        """
                    )


                with d2:

                    st.html(
                        f"""
                        <div class="ai-card">

                            <div style="
                                font-size:32px;
                            ">
                                🔬
                            </div>

                            <div style="
                                color:#94a3b8;
                                font-size:12px;
                                font-weight:700;
                                margin-top:8px;
                            ">
                                MICRO CRACKS
                            </div>

                            <div style="
                                font-size:20px;
                                font-weight:800;
                                margin-top:7px;
                                color:#ff3864;
                            ">
                                {data['crack_status']}
                            </div>

                        </div>
                        """
                    )


                with d3:

                    st.html(
                        f"""
                        <div class="ai-card">

                            <div style="
                                font-size:32px;
                            ">
                                🔥
                            </div>

                            <div style="
                                color:#94a3b8;
                                font-size:12px;
                                font-weight:700;
                                margin-top:8px;
                            ">
                                THERMAL HOTSPOTS
                            </div>

                            <div style="
                                font-size:20px;
                                font-weight:800;
                                margin-top:7px;
                                color:#ff9f1c;
                            ">
                                {data['hotspot_status']}
                            </div>

                        </div>
                        """
                    )


                # =================================================
                # ENERGY LOSS
                # =================================================

                st.markdown(
                    "### ⚡ Estimated Energy Impact"
                )


                l1, l2, l3 = st.columns(3)


                loss_cards = [
                    (
                        l1,
                        "🧹",
                        "Dust / Soiling",
                        data["dust_loss"],
                        "#00e5ff"
                    ),
                    (
                        l2,
                        "🔬",
                        "Structural Cracks",
                        data["crack_loss"],
                        "#ff3864"
                    ),
                    (
                        l3,
                        "🔥",
                        "Thermal Hotspots",
                        data["hotspot_loss"],
                        "#ff9f1c"
                    )
                ]


                for col, icon, title, value, color in loss_cards:

                    with col:

                        st.html(
                            f"""
                            <div class="metric-container">

                                <div style="
                                    font-size:30px;
                                ">
                                    {icon}
                                </div>

                                <div style="
                                    color:#94a3b8;
                                    margin-top:7px;
                                ">
                                    {title}
                                </div>

                                <div style="
                                    font-size:30px;
                                    font-weight:850;
                                    color:{color};
                                    margin-top:5px;
                                ">
                                    {value:.1f}%
                                </div>

                            </div>
                            """
                        )


                # =================================================
                # RECOMMENDATION
                # =================================================

                st.markdown(
                    "### 🧠 AI Recommendation"
                )


                if data["total_detections"] == 0:

                    recommendation_title = (
                        "✓ PANEL CONDITION OPTIMAL"
                    )

                    recommendation_text = (
                        "No significant AI-detected defects "
                        "were found. Continue routine monitoring "
                        "and scheduled cleaning."
                    )

                    recommendation_color = "#00ff9d"

                elif score >= 50:

                    recommendation_title = (
                        "⚠ ATTENTION RECOMMENDED"
                    )

                    recommendation_text = (
                        "Potential surface defects or soiling "
                        "were detected. Consider cleaning and "
                        "performing a physical inspection."
                    )

                    recommendation_color = "#ffd43b"

                else:

                    recommendation_title = (
                        "⚠ CRITICAL MAINTENANCE REQUIRED"
                    )

                    recommendation_text = (
                        "Multiple significant defects were "
                        "detected. A qualified technician "
                        "should inspect the PV module."
                    )

                    recommendation_color = "#ff3864"


                st.html(
                    f"""
                    <div class="ai-card">

                        <div style="
                            color:{recommendation_color};
                            font-size:18px;
                            font-weight:850;
                        ">
                            {recommendation_title}
                        </div>

                        <div style="
                            color:#94a3b8;
                            line-height:1.7;
                            margin-top:10px;
                        ">
                            {recommendation_text}
                        </div>

                    </div>
                    """
                )


                # =================================================
                # SAVE
                # =================================================

                st.markdown("---")


                record = {

                    "Timestamp":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "Filename":
                        uploaded_file.name,

                    "Inspection_Mode":
                        inspection_mode,

                    "Panel_Status":
                        data["panel_status"],

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
                        f"{data['total_estimated_loss']:.1f}%",

                    "Saved_Image_Path":
                        ""
                }


                if st.button(
                    "💾 SAVE ANALYSIS TO HISTORY",
                    key="save_record_btn"
                ):

                    save_inspection_record(
                        record,
                        data["results_img"]
                    )


    # ========================================================
    # MAINTENANCE GUIDE
    # ========================================================

    st.markdown("---")


    st.html(
        """
        <div class="hero">

            <div class="hero-title"
                 style="font-size:28px;">
                🛠️ Solar Panel Care Center
            </div>

            <div class="hero-subtitle">
                Practical maintenance guidance for
                long-term photovoltaic performance.
            </div>

        </div>
        """
    )


    guide1, guide2 = st.columns(2)


    with guide1:

        st.html(
            """
            <div class="report-card">

                <h3 style="
                    color:#00ff9d;
                ">
                    🧼 Owner's Maintenance Guide
                </h3>

                <h4>
                    01 • Routine Cleaning
                </h4>

                <p style="color:#cbd5e1;">
                    Clean panels approximately every
                    2–3 months depending on dust,
                    pollution and local conditions.
                </p>

                <h4>
                    02 • Correct Timing
                </h4>

                <p style="color:#cbd5e1;">
                    Prefer early morning or evening
                    when panels are cooler.
                </p>

                <h4>
                    03 • Water Quality
                </h4>

                <p style="color:#cbd5e1;">
                    Soft or de-ionized water can help
                    reduce mineral spotting.
                </p>

                <h4>
                    04 • Safe Tools
                </h4>

                <p style="color:#cbd5e1;">
                    Use soft brushes and non-abrasive
                    cleaning tools. Avoid harsh chemicals.
                </p>

                <h4>
                    05 • Visual Inspection
                </h4>

                <p style="color:#cbd5e1;">
                    Check for shading, bird droppings,
                    damaged glass and exposed wiring.
                </p>

            </div>
            """
        )


    with guide2:

        st.html(
            """
            <div class="report-card-warning">

                <h3 style="
                    color:#ffd43b;
                ">
                    📞 Service & Support
                </h3>

                <p style="
                    color:#cbd5e1;
                    line-height:1.7;
                ">
                    If the dashboard reports critical
                    defects or thermal anomalies,
                    arrange an inspection by a
                    qualified solar professional.
                </p>

                <hr>

                <h4>
                    🛰️ Super Six Technical Support
                </h4>

                <p style="color:#94a3b8;">
                    Dashboard Software Integration<br>
                    AI Model Recalibration<br>
                    Computer Vision Development
                </p>

                <p style="
                    color:#00e5ff;
                    font-weight:700;
                ">
                    Team: Super Six
                </p>

                <hr>

                <p style="
                    color:#64748b;
                    font-size:12px;
                ">
                    Demo project information.
                    Replace service information with
                    verified contacts before deployment.
                </p>

            </div>
            """
        )


# ============================================================
# ABOUT PAGE
# ============================================================

with tab_about:

    st.html(
        """
        <div class="hero">

            <div class="status-online">
                <span class="status-dot"></span>
                SUPER SIX • AI RESEARCH PROJECT
            </div>

            <div class="hero-title">
                Solar Vision AI
            </div>

            <div class="hero-subtitle">
                Computer vision meets renewable energy
                maintenance. Detect • Analyze • Predict • Maintain.
            </div>

        </div>
        """
    )


    a1, a2, a3, a4 = st.columns(4)


    about_metrics = [

        ("🧠", "4", "AI Models"),

        ("🔍", "3+", "Defect Types"),

        ("⚡", "YOLOv8", "AI Engine"),

        ("☀️", "PV", "Energy Domain")
    ]


    for col, item in zip(
        [a1, a2, a3, a4],
        about_metrics
    ):

        icon, value, label = item

        with col:

            st.html(
                f"""
                <div class="metric-container"
                     style="text-align:center;">

                    <div style="
                        font-size:32px;
                    ">
                        {icon}
                    </div>

                    <div style="
                        font-size:28px;
                        font-weight:850;
                        color:#00e5ff;
                        margin:6px;
                    ">
                        {value}
                    </div>

                    <div style="
                        color:#94a3b8;
                        font-size:13px;
                    ">
                        {label}
                    </div>

                </div>
                """
            )


    st.markdown("")


    c1, c2 = st.columns(2)


    with c1:

        st.html(
            """
            <div class="about-card">

                <h2>
                    👨‍💻 Project Details
                </h2>

                <p style="color:#cbd5e1;">
                    <b>Lead Developer:</b>
                    Abhijeet Singh
                </p>

                <p style="color:#cbd5e1;">
                    <b>Program:</b>
                    B.Tech
                </p>

                <p style="color:#cbd5e1;">
                    <b>Institution:</b>
                    United Institute of Technology
                </p>

                <p style="color:#cbd5e1;">
                    <b>Team:</b>
                    Super Six
                </p>

                <hr>

                <h3>
                    🎯 Core Objectives
                </h3>

                <p style="color:#94a3b8;">
                    • Automate PV maintenance inspection
                </p>

                <p style="color:#94a3b8;">
                    • Detect dust and surface soiling
                </p>

                <p style="color:#94a3b8;">
                    • Identify cracks and physical defects
                </p>

                <p style="color:#94a3b8;">
                    • Detect thermal hotspot anomalies
                </p>

                <p style="color:#94a3b8;">
                    • Estimate potential energy impact
                </p>

            </div>
            """
        )


    with c2:

        st.html(
            """
            <div class="about-card">

                <h2>
                    🛠️ Technology Stack
                </h2>

                <div class="model-pill">
                    <span>🧠 Deep Learning</span>
                    <b style="color:#00e5ff;">
                        YOLOv8
                    </b>
                </div>

                <div class="model-pill">
                    <span>🖥️ Interface</span>
                    <b style="color:#00ff9d;">
                        Streamlit
                    </b>
                </div>

                <div class="model-pill">
                    <span>📷 Computer Vision</span>
                    <b style="color:#ffd43b;">
                        OpenCV
                    </b>
                </div>

                <div class="model-pill">
                    <span>🖼️ Image Processing</span>
                    <b style="color:#a855f7;">
                        PIL
                    </b>
                </div>

                <div class="model-pill">
                    <span>📊 Data Management</span>
                    <b style="color:#00e5ff;">
                        Pandas
                    </b>
                </div>

                <hr>

                <h3>
                    🚀 Future Scope
                </h3>

                <p style="color:#94a3b8;">
                    • Drone-based autonomous inspection
                </p>

                <p style="color:#94a3b8;">
                    • Thermal camera integration
                </p>

                <p style="color:#94a3b8;">
                    • Cloud inspection history
                </p>

                <p style="color:#94a3b8;">
                    • Predictive maintenance
                </p>

                <p style="color:#94a3b8;">
                    • IoT sensor integration
                </p>

            </div>
            """
        )


    st.markdown("")


    st.html(
        """
        <div class="ai-card"
             style="text-align:center;">

            <div style="
                font-size:45px;
            ">
                ☀️
            </div>

            <h2>
                AI for Clean Energy
            </h2>

            <p style="
                color:#94a3b8;
                max-width:750px;
                margin:auto;
                line-height:1.7;
            ">
                Solar Vision AI combines computer vision
                and renewable-energy maintenance workflows
                to make photovoltaic inspection faster,
                smarter and more data-driven.
            </p>

        </div>
        """
    )


# ============================================================
# HISTORY PAGE
# ============================================================

with tab_history:

    st.html(
        """
        <div class="hero">

            <div class="status-online">
                <span class="status-dot"></span>
                INSPECTION DATABASE
            </div>

            <div class="hero-title"
                 style="font-size:34px;">
                📁 Inspection History
            </div>

            <div class="hero-subtitle">
                Review previous AI inspections,
                detection results and analyzed images.
            </div>

        </div>
        """
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

            # =================================================
            # SUMMARY
            # =================================================

            h1, h2, h3, h4 = st.columns(4)


            total_records = len(
                history_df
            )


            if "Detections_Count" in history_df.columns:

                total_defects = pd.to_numeric(
                    history_df[
                        "Detections_Count"
                    ],
                    errors="coerce"
                ).fillna(0).sum()

            else:

                total_defects = 0


            if "Priority" in history_df.columns:

                high_priority = (
                    history_df[
                        history_df[
                            "Priority"
                        ]
                        .astype(str)
                        .str.upper()
                        == "HIGH"
                    ].shape[0]
                )

            else:

                high_priority = 0


            with h1:

                st.html(
                    f"""
                    <div class="metric-container">

                        <small style="
                            color:#94a3b8;
                        ">
                            📁 INSPECTIONS
                        </small>

                        <h2 style="
                            color:#00e5ff;
                        ">
                            {total_records}
                        </h2>

                    </div>
                    """
                )


            with h2:

                st.html(
                    f"""
                    <div class="metric-container">

                        <small style="
                            color:#94a3b8;
                        ">
                            🔎 TOTAL DEFECTS
                        </small>

                        <h2 style="
                            color:#ffd43b;
                        ">
                            {int(total_defects)}
                        </h2>

                    </div>
                    """
                )


            with h3:

                st.html(
                    f"""
                    <div class="metric-container">

                        <small style="
                            color:#94a3b8;
                        ">
                            ⚠️ HIGH PRIORITY
                        </small>

                        <h2 style="
                            color:#ff3864;
                        ">
                            {high_priority}
                        </h2>

                    </div>
                    """
                )


            with h4:

                st.html(
                    """
                    <div class="metric-container">

                        <small style="
                            color:#94a3b8;
                        ">
                            🛰️ SYSTEM
                        </small>

                        <h2 style="
                            color:#00ff9d;
                        ">
                            ONLINE
                        </h2>

                    </div>
                    """
                )


            st.markdown("---")


            # =================================================
            # TABLE
            # =================================================

            st.subheader(
                "📊 Inspection Log"
            )


            preferred_columns = [
                "Timestamp",
                "Filename",
                "Inspection_Mode",
                "Panel_Status",
                "Dust_Status",
                "Crack_Status",
                "Hotspot_Status",
                "Health_Score",
                "Priority",
                "Detections_Count",
                "Estimated_Loss_Pct"
            ]


            display_columns = [
                column
                for column in preferred_columns
                if column in history_df.columns
            ]


            st.dataframe(
                history_df[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True
            )


            st.markdown("---")


            # =================================================
            # GALLERY
            # =================================================

            st.subheader(
                "🖼️ Visual Inspection Gallery"
            )


            for _, row in history_df.iterrows():

                timestamp = row.get(
                    "Timestamp",
                    "Unknown"
                )

                filename = row.get(
                    "Filename",
                    "Unknown"
                )

                health = row.get(
                    "Health_Score",
                    "N/A"
                )

                priority = row.get(
                    "Priority",
                    "N/A"
                )


                title = (
                    f"📌 {timestamp} | "
                    f"{filename} | "
                    f"Health: {health} | "
                    f"Priority: {priority}"
                )


                with st.expander(title):

                    info_col, img_col = (
                        st.columns(2)
                    )


                    with info_col:

                        st.html(
                            f"""
                            <div class="ai-card">

                                <p>
                                    <b>Inspection Mode:</b>
                                    {row.get(
                                        'Inspection_Mode',
                                        'N/A'
                                    )}
                                </p>

                                <p>
                                    <b>Panel:</b>
                                    {row.get(
                                        'Panel_Status',
                                        'N/A'
                                    )}
                                </p>

                                <p>
                                    <b>Dust:</b>
                                    {row.get(
                                        'Dust_Status',
                                        'N/A'
                                    )}
                                </p>

                                <p>
                                    <b>Cracks:</b>
                                    {row.get(
                                        'Crack_Status',
                                        'N/A'
                                    )}
                                </p>

                                <p>
                                    <b>Hotspots:</b>
                                    {row.get(
                                        'Hotspot_Status',
                                        'N/A'
                                    )}
                                </p>

                                <p>
                                    <b>Health Score:</b>
                                    {health}
                                </p>

                                <p>
                                    <b>Estimated Loss:</b>
                                    {row.get(
                                        'Estimated_Loss_Pct',
                                        'N/A'
                                    )}
                                </p>

                                <p>
                                    <b>Priority:</b>
                                    {priority}
                                </p>

                                <p>
                                    <b>Defects:</b>
                                    {row.get(
                                        'Detections_Count',
                                        'N/A'
                                    )}
                                </p>

                            </div>
                            """
                        )


                    with img_col:

                        image_path = str(
                            row.get(
                                "Saved_Image_Path",
                                ""
                            )
                        )


                        if (
                            image_path
                            and image_path != "nan"
                            and os.path.exists(
                                image_path
                            )
                        ):

                            st.image(
                                image_path,
                                caption="AI Analyzed Output",
                                use_container_width=True
                            )

                        else:

                            st.warning(
                                "⚠️ Saved image not found."
                            )


            # =================================================
            # EXPORT / CLEAR
            # =================================================

            st.markdown("---")


            export_col, clear_col = (
                st.columns(2)
            )


            with export_col:

                csv_data = (
                    history_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )


                st.download_button(
                    label="📥 EXPORT INSPECTION CSV",
                    data=csv_data,
                    file_name=(
                        "Solar_Inspection_Report_"
                        f"{datetime.now().strftime('%Y%m%d')}"
                        ".csv"
                    ),
                    mime="text/csv",
                    key="export_csv_btn"
                )


            with clear_col:

                if st.button(
                    "🗑️ CLEAR HISTORY",
                    key="clear_history_btn"
                ):

                    if os.path.exists(
                        CSV_FILE
                    ):

                        os.remove(
                            CSV_FILE
                        )

                    init_data_storage()

                    st.rerun()


        else:

            st.html(
                """
                <div class="ai-card"
                     style="
                        text-align:center;
                        padding:60px;
                     ">

                    <div style="
                        font-size:55px;
                    ">
                        📁
                    </div>

                    <h2>
                        No Inspections Yet
                    </h2>

                    <p style="
                        color:#64748b;
                    ">
                        Run an AI inspection and save
                        the result to create your history.
                    </p>

                </div>
                """
            )

    else:

        st.info(
            "No inspection history found."
        )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        margin-top:70px;
        padding:30px;

        text-align:center;

        border-top:
            1px solid rgba(255,255,255,0.08);

        color:#64748b;
    ">

        <div style="
            font-size:22px;
            font-weight:850;
            color:#00e5ff;
            letter-spacing:2px;
        ">
            ☀️ SOLAR VISION
        </div>

        <div style="
            margin-top:8px;
            font-size:12px;
        ">
            Powered by YOLOv8 • OpenCV • Streamlit • Python
        </div>

        <div style="
            margin-top:8px;
            font-size:11px;
            color:#475569;
        ">
            SUPER SIX • AI FOR CLEAN ENERGY
        </div>

    </div>
    """
)