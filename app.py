import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Student Depression Predictor", page_icon="🧠", layout="wide")

# Custom CSS for UI enhancement
st.markdown("""
    <style>
    .main {
        background-color: #f0f4ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title {
        font-size: 34px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #34495e;
        margin-bottom: 30px;
    }
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 class='title'>🧠 Student Depression Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter the details below to predict the student's mental health risk.</p>", unsafe_allow_html=True)

# Load model
model_path = "AdaBoost_model.pkl"
try:
    model = joblib.load(model_path)
except:
    st.error("Model file not found! Please upload AdaBoost_model.pkl")
    st.stop()

# Input form
with st.form("prediction_form"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### Personal Details")

    col1, col2 = st.columns(2)
    with col1:
        student_id = st.number_input("Student ID", min_value=1, step=1)
        age = st.slider("Age", 15, 60, 20)
        city = st.text_input("City")

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        profession = st.text_input("Profession", "Student")
        degree = st.text_input("Degree", "B.Tech")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### Academic & Lifestyle Details")

    col3, col4 = st.columns(2)
    with col3:
        academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
        study_hours = st.slider("Study Hours per Week", 0, 100, 20)
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)

    with col4:
        sleep_duration = st.selectbox("Sleep Duration", ["<5 hrs", "5-6 hrs", "7-8 hrs", "9+ hrs"])
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])
        work_pressure = st.slider("Work Pressure (1-5)", 1, 5, 2)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### Mental Health Indicators")

    suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["No", "Yes"])
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
    financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)

    st.markdown("</div>", unsafe_allow_html=True)

    submit_btn = st.form_submit_button("Predict Depression Risk")

# On submit
if submit_btn:
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    sleep_map = {"<5 hrs": 4, "5-6 hrs": 5.5, "7-8 hrs": 7.5, "9+ hrs": 9}

    input_data = pd.DataFrame([[
        student_id, gender_map[gender], age, city, profession, academic_pressure,
        work_pressure, cgpa, study_hours, sleep_map[sleep_duration],
        1 if dietary_habits == "Healthy" else 0,
        degree, 1 if suicidal_thoughts == "Yes" else 0, financial_stress,
        1 if family_history == "Yes" else 0
    ]], columns=[
        "id", "Gender", "Age", "City", "Profession", "Academic Pressure",
        "Work Pressure", "CGPA", "Work/Study Hours", "Sleep Duration",
        "Dietary Habits", "Degree", "Have Suicidal Thoughts",
        "Financial Stress", "Family History"
    ])

    prediction = model.predict_proba(input_data)[0][1] * 100

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("## 🎯 Prediction Result")
    st.write(f"### 🔍 Depression Probability: **{prediction:.2f}%**")

    if prediction < 30:
        st.success("Low risk — Keep maintaining a healthy lifestyle! 😊")
    elif prediction < 60:
        st.warning("Moderate risk — Consider balancing academics and mental health. ⚠️")
    else:
        st.error("High risk — Professional support is strongly recommended. 🚨")

    st.markdown("</div>", unsafe_allow_html=True)

