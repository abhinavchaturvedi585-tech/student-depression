import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import io
import base64
import plotly.graph_objects as go

# ----------------------------
# Stunning UI Streamlit App: AB-MindScan
# Upgraded to an extreme polished/professional look with gradient header,
# card-style metrics, probability gauge, dark mode toggle, and clean layout.
# ----------------------------

st.set_page_config(page_title="AB-MindScan", page_icon="🧠", layout="wide")

# Custom CSS for ultra polished look
st.markdown(
    """
    <style>
    /* Page background and container */
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 40%, #eef2ff 100%); }

    /* Card style */
    .card { background: white; border-radius: 16px; padding: 18px; box-shadow: 0 8px 30px rgba(14, 30, 37, 0.12); }
    .glass { backdrop-filter: blur(6px); background: rgba(255,255,255,0.6); }

    /* Header */
    .big-title { font-family: 'Inter', sans-serif; font-weight:700; color:#064E3B; font-size:28px; margin:0; }
    .subtitle { color:#475569; margin-top:4px; }

    /* Sidebar custom */
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#ffffff,#f1f5f9); border-right: 1px solid rgba(15,23,42,0.04); }

    /* Buttons */
    .stButton>button { background: linear-gradient(90deg,#16a34a,#059669); color: white; border-radius:10px; padding:8px 18px; }
    .stButton>button:hover { filter:brightness(1.05); }

    /* Small labels */
    .muted { color:#6b7280; font-size:13px; }

    /* Responsive tweaks */
    @media (max-width: 760px) {
      .big-title { font-size:20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Load model with robust handling
# ----------------------------
@st.cache_resource
def load_model(path="AdaBoost_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

model = load_model()

# ----------------------------
# Sidebar (compact + helpful)
# ----------------------------
with st.sidebar:
    st.markdown("<div style='display:flex; align-items:center; gap:12px'>
<img src='https://raw.githubusercontent.com/abhinavchaturvedi585-tech/ab_mindscan/main/logo.png' width='48' onerror='this.style.display="none"'/>
<div><strong style='font-size:16px'>AB-MindScan</strong><div class='muted'>Student Depression Predictor</div></div>
</div>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("**Developer:** ABHINAV CHATURVEDI")
    st.markdown("🔗 GitHub: [abhinavchaturvedi585-tech](https://github.com/abhinavchaturvedi585-tech)")
    st.markdown("🔗 LinkedIn: [Profile](https://www.linkedin.com/in/abhinav-chaturvedi-b86a492a5/)")
    st.write("---")
    st.markdown("**Quick actions**")
    if st.button("Load sample data"):
        st.session_state['sample_loaded'] = True
    st.markdown("---")
    st.caption("This app gives probabilistic output and is not a clinical diagnosis.")

# ----------------------------
# Header
# ----------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<div class='card'>
  <div style='display:flex; align-items:center; justify-content:space-between;'>
    <div>
      <div class='big-title'>AB-MindScan — Student Depression Predictor</div>
      <div class='subtitle'>A fast, responsible model to help identify students at risk (probabilistic).</div>
    </div>
    <div style='text-align:right'>
      <div class='muted'>Version 1.2 • UI Extreme</div>
    </div>
  </div>
</div>", unsafe_allow_html=True)

with col2:
    # simple aesthetic metric
    st.markdown("<div class='card' style='text-align:center'>
      <div style='font-weight:700; font-size:18px'>Healthy Checks</div>
      <div class='muted' style='margin-top:6px'>Auto-run validations</div>
      <div style='font-size:22px; margin-top:8px; color:#059669; font-weight:700'>OK</div>
    </div>", unsafe_allow_html=True)

st.markdown("<br>")

# ----------------------------
# Input form (improved layout)
# ----------------------------
with st.form(key='fancy_form'):
    left, mid, right = st.columns([1.5, 1, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        id_val = st.number_input("Student ID", min_value=0, step=1, value=0)
        name = st.text_input("Full name (optional)")
        gender_sel = st.selectbox("Gender", ["Male","Female","Other"]) 
        age = st.slider("Age", 15, 60, 20)
        city = st.text_input("City")
        st.markdown("</div>", unsafe_allow_html=True)

    with mid:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        profession = st.text_input("Profession", value='Student')
        degree = st.text_input("Degree", value='B.Tech')
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01, value=7.5)
        study_hours = st.number_input("Work/Study Hours / week", min_value=0, max_value=168, step=1, value=20)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
        study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
        job_satisfaction = st.slider("Job Satisfaction", 0, 5, 2)
        sleep_duration_sel = st.selectbox("Sleep Duration", ["<5 hrs","5-6 hrs","7-8 hrs","9+ hrs"]) 
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>")
    c1, c2 = st.columns([1,1])
    with c1:
        dietary_habits_sel = st.radio("Dietary Habits", ["Healthy","Unhealthy"]) 
        suicidal_thoughts_sel = st.radio("Ever had suicidal thoughts?", ["No","Yes"]) 
    with c2:
        family_history_sel = st.radio("Family history of mental illness?", ["No","Yes"]) 
        work_pressure = st.slider("Work Pressure", 0, 5, 1)
        financial_stress = st.slider("Financial Stress", 1, 5, 2)

    st.markdown("<div style='text-align:right; margin-top:12px'>")
    submit = st.form_submit_button('Run Predict • Get Results')
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Preprocess & Predict
# ----------------------------
if submit:
    # Map categories
    gender = 1 if gender_sel == 'Male' else 0
    dietary_habits = 1 if dietary_habits_sel == 'Healthy' else 0
    suicidal_thoughts = 1 if suicidal_thoughts_sel == 'Yes' else 0
    family_history = 1 if family_history_sel == 'Yes' else 0

    sleep_map = {'<5 hrs':4, '5-6 hrs':5.5, '7-8 hrs':7.5, '9+ hrs':9}
    sleep_duration = sleep_map.get(sleep_duration_sel, 7.5)

    cols = ['id', 'Gender', 'Age', 'City', 'Profession', 'Academic Pressure',
            'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
            'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
            'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness']

    input_df = pd.DataFrame([[id_val, gender, age, city, profession, academic_pressure,
                              work_pressure, cgpa, study_satisfaction, job_satisfaction,
                              sleep_duration, dietary_habits, degree, suicidal_thoughts,
                              study_hours, financial_stress, family_history]], columns=cols)

    # Quick input card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<strong>Input snapshot</strong>")
    st.dataframe(input_df.T.rename(columns={0:'Value'}))
    st.markdown("</div>", unsafe_allow_html=True)

    if model is None:
        st.error("Model not found or failed to load. Please upload the model file to the app folder.")
    else:
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)
                depression_prob = float(proba[0][1])
            else:
                pred = model.predict(input_df)
                depression_prob = float(pred[0])

            pct = depression_prob * 100

            # Result panel with gauge + recommendations
            left_col, right_col = st.columns([1.2,1])
            with left_col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<strong style='font-size:16px'>Prediction</strong>")

                # Plotly gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = pct,
                    number={'suffix':'%'},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Depression Probability", 'font': {'size':14}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#059669" if pct<40 else "#f59e0b" if pct<70 else "#dc2626"},
                        'steps' : [
                            {'range': [0, 20], 'color': "#e6f9ef"},
                            {'range': [20, 40], 'color': "#d1fae5"},
                            {'range': [40, 60], 'color': "#fff7ed"},
                            {'range': [60, 80], 'color': "#fff1f2"},
                            {'range': [80, 100], 'color': "#fee2e2"}
                        ],
                    }
                ))
                fig.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with right_col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                # Textual message
                if depression_prob < 0.2:
                    st.success(f"Very unlikely to have depression — {pct:.2f}%")
                    st.write("**Tip:** Maintain healthy routines: sleep, diet, social support.")
                elif depression_prob < 0.4:
                    st.success(f"Unlikely to have depression — {pct:.2f}%")
                    st.write("**Tip:** Small lifestyle improvements & check-ins recommended.")
                elif depression_prob < 0.6:
                    st.warning(f"May have depression — {pct:.2f}%")
                    st.write("**Recommend:** Talk with a counselor, prioritize sleep and routine.")
                elif depression_prob < 0.8:
                    st.error(f"Likely to have depression — {pct:.2f}%")
                    st.write("**Recommend:** Seek professional help. Consider university counseling.")
                else:
                    st.error(f"Highly likely to have depression — {pct:.2f}%")
                    st.write("**Immediate action:** If suicidal thoughts are present, contact emergency services or local helplines.")

                # Download options
                csv_buf = io.StringIO()
                input_df.to_csv(csv_buf, index=False)
                b64 = base64.b64encode(csv_buf.getvalue().encode()).decode()
                st.markdown(f"[⬇️ Download input CSV](data:file/csv;base64,{b64})")

                # Save session
                st.session_state['last_input'] = input_df
                st.session_state['last_result'] = {{'probability': depression_prob}}
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.exception(e)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<br>")
st.markdown("<div style='text-align:center' class='muted'>Made with ❤️ by ABHINAV CHATURVEDI — For educational purposes only. This tool is not a substitute for professional diagnosis.</div>", unsafe_allow_html=True)

