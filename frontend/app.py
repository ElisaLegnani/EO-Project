import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #e5edf7;
    }

    [data-testid="stSidebar"] {
        background-color: #111c2e;
    }

    .hero {
        padding: 1.5rem 0 1rem 0;
    }

    .hero h1 {
        color: #f1f5f9;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }

    .hero p {
        color: #9caec4;
        font-size: 1.05rem;
    }

    .result-card {
        background: #15243a;
        border: 1px solid #263b56;
        border-radius: 14px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }

    div.stButton > button {
        background-color: #2b83d5;
        color: white;
        border: 0;
        border-radius: 9px;
        padding: 0.6rem 1.2rem;
    }

    div.stButton > button:hover {
        background-color: #176bb8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Sidebar navigation container */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 10px;
    }

    /* Hide circular radio indicators */
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none;
    }

    /* Navigation buttons */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #15243a;
        border: 1px solid #263b56;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 5px;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    /* Navigation text */
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #9caec4;
        font-size: 15px;
        font-weight: 500;
    }

    /* Hover effect */
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #203652;
        border-color: #2b83d5;
    }

    /* Selected navigation button */
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #1b4d7a;
        border: 1px solid #2b83d5;
    }

    /* Selected text */
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🛰️  Satellite Image Classifier")
    st.caption("Earth observation · Land-cover · CNN")

    page = st.radio("NAVIGATION", ["Classify", "History", "About"])

    st.divider()
    st.caption("Powered by PyTorch, FastAPI and Streamlit")


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Satellite Image Classifier</h1>
        <p>
            Explore land-cover predictions from RGB satellite imagery
            using your EuroSAT classification model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Classification page
# --------------------------------------------------

if page == "Classify":
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Upload satellite image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_file:
            image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")

            st.image(
                image,
                caption=uploaded_file.name,
                width="stretch",
            )

            st.caption(f"Image dimensions: {image.width} × {image.height}")

            classify = st.button(
                "Run classification",
                width="stretch",
            )

            if classify:
                with st.spinner("Analysing satellite image..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type,
                                )
                            },
                            timeout=60,
                        )

                        response.raise_for_status()
                        result = response.json()

                        history_entry = {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "filename": uploaded_file.name,
                            "predicted_class": result["predicted_class"],
                            "confidence": result["confidence"],
                            "probabilities": result["probabilities"],
                        }

                        st.session_state.history.insert(
                            0,
                            history_entry,
                        )

                        st.session_state.latest_result = result
                        st.session_state.latest_filename = uploaded_file.name

                    except requests.RequestException as error:
                        st.error(f"Could not reach the prediction API: {error}")

        else:
            st.info("Upload an image to get started.")

    # Results panel

    with right:
        st.subheader("Prediction results")

        if "latest_result" in st.session_state:
            result = st.session_state.latest_result

            st.markdown(
                f"""
                <div class="result-card">
                    <p style="color:#9caec4;margin-bottom:5px;">
                        PREDICTED LAND-COVER CLASS
                    </p>
                    <h2 style="color:#63b3ed;margin-top:0;">
                        {result["predicted_class"]}
                    </h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric(
                "Model confidence",
                f"{result['confidence'] * 100:.2f}%",
            )

            st.markdown("### Class probabilities")

            probability_data = pd.DataFrame(
                {
                    "Class": list(result["probabilities"].keys()),
                    "Probability": [
                        value * 100 for value in result["probabilities"].values()
                    ],
                }
            ).sort_values(
                "Probability",
                ascending=False,
            )

            st.bar_chart(
                probability_data.set_index("Class"),
                horizontal=True,
            )

            st.dataframe(
                probability_data.style.format({"Probability": "{:.2f}%"}),
                width="stretch",
                hide_index=True,
            )

        else:
            st.markdown(
                """
                <div class="result-card">
                    <h3 style="color:#9caec4;">
                        Waiting for an image
                    </h3>
                    <p style="color:#8497ae;">
                        Your prediction and class probabilities
                        will appear here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------
# History page
# --------------------------------------------------

elif page == "History":
    st.subheader("Prediction history")

    history = st.session_state.history

    if history:
        history_table = pd.DataFrame(
            [
                {
                    "Time": item["time"],
                    "Image": item["filename"],
                    "Predicted class": item["predicted_class"],
                    "Confidence": f"{item['confidence'] * 100:.2f}%",
                }
                for item in history
            ]
        )

        st.dataframe(
            history_table,
            width="stretch",
            hide_index=True,
        )

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No predictions yet. Classify an image to see it here.")


# --------------------------------------------------
# About page
# --------------------------------------------------

elif page == "About":
    st.subheader("About this project")

    st.markdown(
        """
        [![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/ElisaLegnani/EO-Project/)

        This application uses a deep learning model to classify
        satellite images into ten land-cover categories.

        **Dataset:** EuroSAT Sentinel-2 RGB

        **Input:** RGB satellite imagery

        **Image preprocessing:** Resize to 64 × 64 pixels

        **Model:** PyTorch image classifier

        **Backend:** FastAPI

        **Frontend:** Streamlit

        ### What does the model predict?

        - AnnualCrop
        - Forest
        - HerbaceousVegetation
        - Highway
        - Industrial
        - Pasture
        - PermanentCrop
        - Residential
        - River
        - SeaLake

        ### How it works

        1. Upload an RGB satellite image.
        2. The API preprocesses the image.
        3. The trained PyTorch model predicts its class.
        4. The application displays the prediction and probabilities.

        **Note:** Model confidence is not a guarantee that a prediction
        is correct.
        """
    )
