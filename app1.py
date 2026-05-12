# app.py

import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="JANU'S Hospital AI System",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

.block-container {
    padding-top: 1rem;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #0072ff, #00c6ff);
    color: white;
}

.stTextInput>div>div>input,
.stNumberInput>div>div>input {
    border-radius: 10px;
    border: 1px solid #00c6ff;
    background-color: #f9f9f9;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<h1 style='text-align: center;'>
🏥 JANU'S Hospital
</h1>
<h3 style='text-align: center; color: #d1d5db;'>
AI-Based Breast Cancer Prediction System
</h3>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------
@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        "BC_model.h5",
        compile=False
    )

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

try:
    model, scaler = load_artifacts()

except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🏥 Navigation")

page = st.sidebar.radio(
    "Select Option",
    ["Prediction System", "About"]
)

# ---------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------
if page == "About":

    st.subheader("ℹ About Project")

    st.write("""
    This AI system predicts whether a breast tumor is:

    - 🟢 Benign
    - 🔴 Malignant

    using Deep Learning with TensorFlow.

    The prediction is based on 30 tumor diagnostic features
    from the Breast Cancer Wisconsin Dataset.
    """)

# ---------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------
if page == "Prediction System":

    st.subheader("🧬 Input Tumor Features")

    col1, col2, col3 = st.columns(3)

    # ---------------------------------------------------
    # COLUMN 1
    # ---------------------------------------------------
    with col1:

        radius_mean = st.number_input(
            'radius_mean',
            value=14.0
        )

        texture_mean = st.number_input(
            'texture_mean',
            value=20.0
        )

        perimeter_mean = st.number_input(
            'perimeter_mean',
            value=90.0
        )

        area_mean = st.number_input(
            'area_mean',
            value=600.0
        )

        smoothness_mean = st.number_input(
            'smoothness_mean',
            value=0.1
        )

        compactness_mean = st.number_input(
            'compactness_mean',
            value=0.15
        )

        concavity_mean = st.number_input(
            'concavity_mean',
            value=0.2
        )

        concave_points_mean = st.number_input(
            'concave points_mean',
            value=0.1
        )

        symmetry_mean = st.number_input(
            'symmetry_mean',
            value=0.2
        )

        fractal_dimension_mean = st.number_input(
            'fractal_dimension_mean',
            value=0.06
        )

    # ---------------------------------------------------
    # COLUMN 2
    # ---------------------------------------------------
    with col2:

        radius_se = st.number_input(
            'radius_se',
            value=0.2
        )

        texture_se = st.number_input(
            'texture_se',
            value=1.0
        )

        perimeter_se = st.number_input(
            'perimeter_se',
            value=1.5
        )

        area_se = st.number_input(
            'area_se',
            value=20.0
        )

        smoothness_se = st.number_input(
            'smoothness_se',
            value=0.005
        )

        compactness_se = st.number_input(
            'compactness_se',
            value=0.02
        )

        concavity_se = st.number_input(
            'concavity_se',
            value=0.03
        )

        concave_points_se = st.number_input(
            'concave points_se',
            value=0.01
        )

        symmetry_se = st.number_input(
            'symmetry_se',
            value=0.03
        )

        fractal_dimension_se = st.number_input(
            'fractal_dimension_se',
            value=0.004
        )

    # ---------------------------------------------------
    # COLUMN 3
    # ---------------------------------------------------
    with col3:

        radius_worst = st.number_input(
            'radius_worst',
            value=16.0
        )

        texture_worst = st.number_input(
            'texture_worst',
            value=25.0
        )

        perimeter_worst = st.number_input(
            'perimeter_worst',
            value=105.0
        )

        area_worst = st.number_input(
            'area_worst',
            value=800.0
        )

        smoothness_worst = st.number_input(
            'smoothness_worst',
            value=0.12
        )

        compactness_worst = st.number_input(
            'compactness_worst',
            value=0.2
        )

        concavity_worst = st.number_input(
            'concavity_worst',
            value=0.3
        )

        concave_points_worst = st.number_input(
            'concave points_worst',
            value=0.15
        )

        symmetry_worst = st.number_input(
            'symmetry_worst',
            value=0.25
        )

        fractal_dimension_worst = st.number_input(
            'fractal_dimension_worst',
            value=0.08
        )

    # ---------------------------------------------------
    # FINAL INPUT DICTIONARY
    # ---------------------------------------------------
    input_data = {

        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,

        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,

        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst,
    }

    st.divider()

    # ---------------------------------------------------
    # PREDICTION BUTTON
    # ---------------------------------------------------
    if st.button("🔍 Predict Cancer Type"):

        input_df = pd.DataFrame([input_data])

        # ---------------------------------------------------
        # SAFETY CHECK
        # ---------------------------------------------------
        try:

            if list(input_df.columns) != list(scaler.feature_names_in_):

                st.error(
                    "Feature mismatch between input and trained scaler."
                )

                st.stop()

        except:
            pass

        # ---------------------------------------------------
        # SCALE INPUT
        # ---------------------------------------------------
        input_scaled = scaler.transform(input_df)

        # ---------------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------------
        prediction = float(
            model.predict(
                input_scaled,
                verbose=0
            )[0][0]
        )

        # ---------------------------------------------------
        # CLASS LABEL
        # ---------------------------------------------------
        predicted_class = (
            "🔴 Malignant"
            if prediction > 0.5
            else "🟢 Benign"
        )

        confidence = prediction * 100

        # ---------------------------------------------------
        # RESULT CARD
        # ---------------------------------------------------
        st.markdown("""
        <br>
        """, unsafe_allow_html=True)

        st.success("Prediction Completed Successfully")

        st.markdown(f"""
        <div style="
            background-color:#111827;
            padding:30px;
            border-radius:20px;
            text-align:center;
            border:2px solid #00c6ff;
        ">
            <h2>{predicted_class}</h2>
            <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------
        # PROBABILITY BAR
        # ---------------------------------------------------
        st.subheader("Prediction Probability")

        st.progress(float(prediction))

        st.write(f"Raw Prediction Value: {prediction:.4f}")

        # ---------------------------------------------------
        # INFO MESSAGE
        # ---------------------------------------------------
        if prediction > 0.5:

            st.error("""
            The model predicts a higher probability
            of malignant tumor characteristics.
            Please consult a medical professional.
            """)

        else:

            st.success("""
            The model predicts benign tumor characteristics.
            Still consult a medical professional for confirmation.
            """)










