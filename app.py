import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import base64

# ----------------------------
# App meta & page config
# ----------------------------
st.set_page_config(page_title="AB-MindScan — Improved UI", page_icon="🧠✨", layout="wide")

# ----------------------------
# Styling (modern card-like look)
# ----------------------------
st.markdown(
    """
    <style>
    .main {background-color: #f7fbf8}
    .stApp > header {background: linear-gradient(90deg, #2E8B57, #2FAF7A);}
    .card {background: white; border-radius: 14px; padding: 18px; box-shadow: 0 6px 18px rgba(46,139,87,0.08);} 
    .muted {color: #6b7280;}
    .big-num {font-size: 28px; font-weight:700}
    .small {font-size:12px; color:#6b7280}
    .hero-title {text-align:center; font-size:34px; font-weight:800; color:#0f5132}
    .hero-sub {text-align:center; color:#4b5563}
    .center {display:flex; justify-content:center}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helper: load model & image
# ----------------------------
@st.cache_resource
def load_model(path="AdaBoost_model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_image(path):
    try:
        return Image.open(path)
    except Exception:
        return None

model = load_model()
hero_img = load_image("Artificial Intelligence Application in Mental Health Research copy.jpg")

# ----------------------------
# Sidebar (compact + useful)
# ----------------------------
with st.sidebar:
    st.markdown("# 🧠 AB-MindScan")
    st.markdown("Student Depression Predictor — friendly, lightweight, explainable UI")
    st.markdown("---")
    st.markdown("**Developer**: Abhinav Chaturvedi")
    st.markdown("[GitHub](https://github.com/abhinavchaturvedi585-tech) • [LinkedIn](https://linkedin.com/in/abhinav-chaturvedi-b86a492a5)")
    st.markdown("---")
    st.markdown("**Model status**")
    if model is None:
        st.error("Model not found. Make sure `AdaBoost_model.pkl` is in the app folder.")
    else:
        st.success("Model loaded ✅")
    st.markdown("---")
    st.markdown("**Quick tips**")
    st.write("• Fill all fields inside the form and click Predict.\n• Use realistic numeric ranges for best results.")
    st.markdown("---")
    st.caption("Built for educational/demo purposes. Not a clinical diagnosis tool.")

# ----------------------------
# Header / Hero
# ----------------------------
st.markdown("<div class='hero-title'>Student Depression Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Enter student lifestyle & academic details — get a probability & simple recommendation</div>", unsafe_allow_html=True)
if hero_img is not None:
    st.image(hero_img, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Input Form (use st.form so users can edit before submit)
# ----------------------------
with st.form(key="input_form"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        id_val = st.number_input("Student ID", min_value=0, step=1, help="Any numeric identifier")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
        age = st.number_input("Age", min_value=10, max_value=100, step=1, value=20)
        city = st.text_input("City", placeholder="e.g. Bhopal")
        degree = st.text_input("Degree", placeholder="e.g. B.Tech - CS")

    with col2:
        profession = st.text_input("Profession", placeholder="Student / Intern / Part-time job")
        cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, step=0.01, format="%.2f")
        study_hours = st.number_input("Work / Study Hours per day", min_value=0, max_value=24, step=1, value=5)
        sleep_duration = st.selectbox("Average Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"], index=2)

    with col3:
        academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
        study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
        work_pressure = st.slider("Work Pressure", 0, 5, 0)
        job_satisfaction = st.slider("Job Satisfaction", 0, 5, 0)
        financial_stress = st.slider("Financial Stress", 1, 5, 3)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    col4, col5 = st.columns([1,1])
    with col4:
        dietary_habits = st.radio("Dietary Habits", ["Healthy", "Unhealthy"], index=0)
        suicidal_thoughts = st.radio("Ever had suicidal thoughts?", ["No", "Yes"], index=0)
        family_history = st.radio("Family history of mental illness?", ["No", "Yes"], index=0)

    with col5:
        submit = st.form_submit_button("Predict — Analyze")

# ----------------------------
# Preprocess categorical values
# ----------------------------
def map_inputs():
    g = 1 if gender == "Male" else (0 if gender == "Female" else 2)
    diet = 1 if dietary_habits == "Healthy" else 0
    suicidal = 1 if suicidal_thoughts == "Yes" else 0
    fam = 1 if family_history == "Yes" else 0
    sleep_map = {
        'Less than 5 hours': 4,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        'More than 8 hours': 9
    }
    sleep = sleep_map.get(sleep_duration, 7.5)

    # Build dataframe using same column order expected by model
    cols = ['id', 'Gender', 'Age', 'City', 'Profession', 'Academic Pressure',
            'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
            'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
            'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness']

    row = [id_val, g, age, city, profession, academic_pressure,
           work_pressure, cgpa, study_satisfaction, job_satisfaction,
           sleep, diet, degree, suicidal, study_hours, financial_stress, fam]

    return pd.DataFrame([row], columns=cols)

# ----------------------------
# Prediction and results
# ----------------------------
if submit:
    input_df = map_inputs()

    col_res1, col_res2 = st.columns([1,1])

    with col_res1:
        st.markdown("### Result")

        if model is None:
            st.error("Prediction unavailable — model failed to load. Check logs and restart the app.")
        else:
            try:
                proba = model.predict_proba(input_df)[0][1]
            except Exception as e:
                st.error(f"Model inference error: {e}")
                proba = None

            if proba is not None:
                percent = proba * 100
                # Show a nice metric
                st.metric(label="Depression likelihood", value=f"{percent:.2f}%")

                # colour-coded suggestion
                if proba < 0.2:
                    st.success("Very unlikely to have depression — normal range.")
                elif proba < 0.4:
                    st.info("Unlikely — consider simple self-care and monitoring.")
                elif proba < 0.6:
                    st.warning("Some risk — suggest talking to a counselor or trusted person.")
                elif proba < 0.8:
                    st.warning("Likely — please consult a mental health professional soon.")
                else:
                    st.error("High likelihood — seek professional help immediately.")

                # Visual progress bar to represent probability
                progress_val = int(percent)
                st.progress(min(max(progress_val, 0), 100))

                # Quick explanation block
                with st.expander("Why this prediction? (simple explanation)"):
                    st.write("Model uses student profile features like sleep, academic pressure, CGPA and history to predict risk. This is a statistical model and not a diagnosis.")

                # Provide option to download input + result
                result_df = input_df.copy()
                result_df['pred_probability'] = proba
                csv = result_df.to_csv(index=False).encode('utf-8')
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="prediction_result.csv">Download input & result as CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

    with col_res2:
        st.markdown("### Input preview")
        st.dataframe(map_inputs())

        st.markdown("---")
        st.markdown("**Notes & Limitations**")
        st.write("• This demo is for educational use only.\n• Not a substitute for clinical assessment.\n• For troubling thoughts, contact emergency services or a mental health professional.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Made with ❤️ by Abhinav — For demo & learning purposes.")

