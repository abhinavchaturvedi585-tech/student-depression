import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import io
import base64

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="AB-MindScan", page_icon="🧠", layout="wide")

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model(path="AdaBoost_model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

model = load_model()

# ----------------------------
# Sidebar (polished)
# ----------------------------
with st.sidebar:
    st.header("🧠 AB-MindScan")
    st.markdown("**Student Depression Predictor — Quick & Responsible**")
    st.write("---")
    st.caption("*This tool provides a probabilistic indication, not a clinical diagnosis.*")
    st.write("\n")
    st.markdown("**Developer**: ABHINAV CHATURVEDI")
    st.markdown("🔗 GitHub: [abhinavchaturvedi585-tech](https://github.com/abhinavchaturvedi585-tech)")
    st.markdown("🔗 LinkedIn: [Profile](https://www.linkedin.com/in/abhinav-chaturvedi-b86a492a5/)")
    st.write("---")
    st.markdown("**Resources & Help**:")
    st.write("- If someone is in immediate danger, call local emergency services.")
    st.write("- Suicide prevention: 988 (where available) or local helplines.")
    st.write("---")
    st.caption("Version: 1.1 — UI upgrade")

# ----------------------------
# Header with image
# ----------------------------
cols = st.columns([1, 3, 1])
with cols[0]:
    try:
        logo = Image.open('logo.png')
        st.image(logo, width=80)
    except Exception:
        st.write("")
with cols[1]:
    st.markdown("<h1 style='text-align:center; color:#2E8B57;'>AB-MindScan — Student Depression Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: #6b7280;'>Enter a student's lifestyle & habits to get a probabilistic assessment.</p>", unsafe_allow_html=True)
with cols[2]:
    st.write("")

# optional banner image
try:
    banner = Image.open('Artificial Intelligence Application in Mental Health Research copy.jpg')
    st.image(banner, use_column_width=True)
except Exception:
    pass

st.markdown("---")

# ----------------------------
# Input Form (responsive, grouped)
# ----------------------------
with st.form(key='input_form', clear_on_submit=False):
    st.subheader("Student Details")
    c1, c2, c3 = st.columns([1,1,1])

    with c1:
        id_val = st.number_input("Student ID (numeric)", min_value=0, step=1, value=0)
        gender_sel = st.radio("Gender", ["Male","Female"], index=0, horizontal=True)
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=20)
        city = st.text_input("City", placeholder="e.g., Indore")

    with c2:
        profession = st.text_input("Profession", placeholder="e.g., Student")
        degree = st.text_input("Degree", placeholder="e.g., B.Tech")
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01, value=7.5)
        study_hours = st.number_input("Work/Study Hours per week", min_value=0, max_value=168, step=1, value=20)

    with c3:
        academic_pressure = st.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0)
        study_satisfaction = st.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0)
        job_satisfaction = st.slider("Job Satisfaction (1-5)", 0.0, 5.0, 0.0)
        sleep_duration_sel = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]) 

    st.markdown("---")
    c4, c5 = st.columns([1,1])
    with c4:
        dietary_habits_sel = st.radio("Dietary Habits", ["Healthy","Unhealthy"], index=0, horizontal=True)
        suicidal_thoughts_sel = st.radio("Ever had suicidal thoughts?", ["Yes","No"], index=1, horizontal=True)
        family_history_sel = st.radio("Family history of mental illness?", ["Yes","No"], index=1, horizontal=True)
    with c5:
        work_pressure = st.slider("Work Pressure (1-5)", 0.0, 5.0, 0.0)
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)

    submit_btn = st.form_submit_button(label='Predict', help='Click to run prediction')

# ----------------------------
# Preprocess inputs
# ----------------------------
if submit_btn:
    # Map categorical
    gender = 1 if gender_sel == 'Male' else 0
    dietary_habits = 1 if dietary_habits_sel == 'Healthy' else 0
    suicidal_thoughts = 1 if suicidal_thoughts_sel == 'Yes' else 0
    family_history = 1 if family_history_sel == 'Yes' else 0

    sleep_mapping = {
        'Less than 5 hours': 4,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        'More than 8 hours': 9
    }
    sleep_duration = sleep_mapping.get(sleep_duration_sel, 7.5)

    # Build input df (keep same column order expected by your model)
    columns = ['id', 'Gender', 'Age', 'City', 'Profession', 'Academic Pressure',
               'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
               'Sleep Duration', 'Dietary Habits', 'Degree',
               'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
               'Financial Stress', 'Family History of Mental Illness']

    input_df = pd.DataFrame([[id_val, gender, age, city, profession, academic_pressure,
                              work_pressure, cgpa, study_satisfaction, job_satisfaction,
                              sleep_duration, dietary_habits, degree, suicidal_thoughts,
                              study_hours, financial_stress, family_history]],
                            columns=columns)

    # Show a compact preview of the input
    with st.expander("Preview input data (click to expand)", expanded=False):
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))

    # Run prediction
    if model is None:
        st.error("Model is not loaded. Please check the model file on the server.")
    else:
        with st.spinner("Predicting..."):
            try:
                # if model supports predict_proba
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df)
                    depression_prob = float(proba[0][1])
                else:
                    pred = model.predict(input_df)
                    depression_prob = float(pred[0])

                pct = depression_prob * 100

                # Nice visual: progress + message
                st.markdown("### Result")
                progress_bar = st.progress(0)
                for i in range(0, int(pct)+1, max(1, int(pct//10) if pct>=10 else 1)):
                    progress_bar.progress(min(i/100.0, 1.0))

                # Conditional messaging + actionable suggestions
                if depression_prob < 0.2:
                    st.success(f"Very unlikely to have depression — {pct:.2f}%")
                    st.info("Keep monitoring habits and encourage healthy routines.")
                elif 0.2 <= depression_prob < 0.4:
                    st.success(f"Unlikely to have depression — {pct:.2f}%")
                    st.info("Consider small lifestyle changes and regular check-ins.")
                elif 0.4 <= depression_prob < 0.6:
                    st.warning(f"May have depression — {pct:.2f}%")
                    st.write("Recommend: talk with a counselor, maintain sleep/diet routines.")
                elif 0.6 <= depression_prob < 0.8:
                    st.error(f"Likely to have depression — {pct:.2f}%")
                    st.write("Recommend: seek professional help, contact campus counseling.")
                else:
                    st.error(f"Highly likely to have depression — {pct:.2f}%")
                    st.write("If suicidal thoughts are present, seek immediate help and contact emergency services.")

                # Save last result to session for download
                st.session_state['last_input'] = input_df
                st.session_state['last_result'] = {'probability': depression_prob}

                # Download buttons (input + result)
                csv_buf = io.StringIO()
                input_df.to_csv(csv_buf, index=False)
                b64 = base64.b64encode(csv_buf.getvalue().encode()).decode()
                href = f"data:file/csv;base64,{b64}"
                st.markdown(f"[Download input as CSV]({href})")

            except Exception as e:
                st.exception(e)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Made with ❤️ by ABHINAV CHATURVEDI — For educational purposes only.")





