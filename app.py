# app.py
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import io
import base64
import plotly.graph_objects as go

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AB-MindScan", page_icon="🧠", layout="wide")

# ----------------------------
# Custom CSS (use triple quotes to avoid unterminated string issues)
# ----------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 40%, #eef2ff 100%); }
    .card { background: white; border-radius: 14px; padding: 16px; box-shadow: 0 8px 30px rgba(14, 30, 37, 0.08); }
    .muted { color:#6b7280; font-size:13px; }
    .title-large { font-family: Inter, sans-serif; font-weight:700; color:#064E3B; font-size:24px; margin:0; }
    .subtitle { color:#475569; margin-top:4px; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#ffffff,#f1f5f9); border-right: 1px solid rgba(15,23,42,0.04); }
    .stButton>button { background: linear-gradient(90deg,#16a34a,#059669); color: white; border-radius:10px; padding:8px 18px; }
    .stButton>button:hover { filter:brightness(1.03); }
    @media (max-width: 760px) { .title-large { font-size:18px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Model loader
# ----------------------------
@st.cache_resource
def load_model(path: str = "AdaBoost_model.pkl"):
    try:
        mdl = joblib.load(path)
        return mdl
    except Exception as e:
        # return None if not found; show message later in UI
        return None

model = load_model()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px">
            <img src="https://raw.githubusercontent.com/abhinavchaturvedi585-tech/ab_mindscan/main/logo.png"
                 width="48" onerror="this.style.display='none'"/>
            <div>
                <strong style="font-size:16px">AB-MindScan</strong>
                <div class="muted">Student Depression Predictor</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("---")
    st.markdown("**Developer:** ABHINAV CHATURVEDI")
    st.markdown("🔗 GitHub: [abhinavchaturvedi585-tech](https://github.com/abhinavchaturvedi585-tech)")
    st.markdown("🔗 LinkedIn: [Profile](https://www.linkedin.com/in/abhinav-chaturvedi-b86a492a5/)")
    st.write("---")
    st.markdown("**Resources & Help**")
    st.caption("If someone is in immediate danger, contact local emergency services.")
    st.write("---")
    if st.button("Load sample values"):
        st.session_state['load_demo'] = True
    st.caption("This tool gives probabilistic output and is NOT a clinical diagnosis.")

# ----------------------------
# Header
# ----------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        """
        <div class="card">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div>
                    <div class="title-large">AB-MindScan — Student Depression Predictor</div>
                    <div class="subtitle">Probabilistic screening to help identify students at risk.</div>
                </div>
                <div style="text-align:right">
                    <div class="muted">Version 1.2 • UI Extreme</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card" style="text-align:center">
            <div style="font-weight:700; font-size:16px">Health Check</div>
            <div class="muted" style="margin-top:6px">Model & UI status</div>
            <div style="font-size:20px; margin-top:8px; color:#059669; font-weight:700">
        """
        + ("OK" if model is not None else "Model Missing")
        + "</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Input Form
# ----------------------------
with st.form(key="input_form"):
    left, mid, right = st.columns([1.5, 1, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        id_val = st.number_input("Student ID", min_value=0, step=1, value=0)
        name = st.text_input("Full name (optional)")
        gender_sel = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 15, 60, 20)
        city = st.text_input("City")
        st.markdown("</div>", unsafe_allow_html=True)

    with mid:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        profession = st.text_input("Profession", value="Student")
        degree = st.text_input("Degree", value="B.Tech")
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01, value=7.5)
        study_hours = st.number_input("Work/Study Hours per week", min_value=0, max_value=168, step=1, value=20)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
        study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
        job_satisfaction = st.slider("Job Satisfaction", 0, 5, 2)
        sleep_duration_sel = st.selectbox("Sleep Duration", ["<5 hrs", "5-6 hrs", "7-8 hrs", "9+ hrs"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        dietary_habits_sel = st.radio("Dietary Habits", ["Healthy", "Unhealthy"])
        suicidal_thoughts_sel = st.radio("Ever had suicidal thoughts?", ["No", "Yes"])
    with c2:
        family_history_sel = st.radio("Family history of mental illness?", ["No", "Yes"])
        work_pressure = st.slider("Work Pressure", 0, 5, 1)
        financial_stress = st.slider("Financial Stress", 1, 5, 2)

    submit = st.form_submit_button("Run Predict • Get Results")

# If user clicked Load sample values button earlier, prefill some data (optional)
if st.session_state.get("load_demo", False):
    id_val = 123
    gender_sel = "Male"
    age = 21
    city = "Indore"
    profession = "Student"
    degree = "B.Tech"
    cgpa = 7.8
    study_hours = 25
    academic_pressure = 4
    study_satisfaction = 2
    job_satisfaction = 1
    sleep_duration_sel = "5-6 hrs"
    dietary_habits_sel = "Unhealthy"
    suicidal_thoughts_sel = "No"
    family_history_sel = "No"
    work_pressure = 3
    financial_stress = 3

# ----------------------------
# Predict & Display
# ----------------------------
if submit:
    # map categories
    gender = 1 if gender_sel == "Male" else 0
    dietary_habits = 1 if dietary_habits_sel == "Healthy" else 0
    suicidal_thoughts = 1 if suicidal_thoughts_sel == "Yes" else 0
    family_history = 1 if family_history_sel == "Yes" else 0

    sleep_map = {"<5 hrs": 4, "5-6 hrs": 5.5, "7-8 hrs": 7.5, "9+ hrs": 9}
    sleep_duration = sleep_map.get(sleep_duration_sel, 7.5)

    cols = [
        "id", "Gender", "Age", "City", "Profession", "Academic Pressure",
        "Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction",
        "Sleep Duration", "Dietary Habits", "Degree", "Have you ever had suicidal thoughts ?",
        "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"
    ]
    input_df = pd.DataFrame([[
        id_val, gender, age, city, profession, academic_pressure,
        work_pressure, cgpa, study_satisfaction, job_satisfaction,
        sleep_duration, dietary_habits, degree, suicidal_thoughts,
        study_hours, financial_stress, family_history
    ]], columns=cols)

    # show input snapshot
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Input snapshot**")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))
    st.markdown("</div>", unsafe_allow_html=True)

    # model check
    if model is None:
        st.error("Model not found or failed to load. Please upload AdaBoost_model.pkl to the app folder.")
    else:
        try:
            with st.spinner("Predicting..."):
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)
                    depression_prob = float(proba[0][1])
                else:
                    pred = model.predict(input_df)
                    depression_prob = float(pred[0])

            pct = depression_prob * 100.0

            # Result layout
            left_col, right_col = st.columns([1.3, 1])
            with left_col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**Prediction**")

                # Plotly gauge
                color_bar = "#059669" if pct < 40 else "#f59e0b" if pct < 70 else "#dc2626"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pct,
                    number={'suffix': '%', 'font': {'size': 20}},
                    title={'text': "Depression Probability", 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color_bar},
                        'steps': [
                            {'range': [0, 20], 'color': "#e6f9ef"},
                            {'range': [20, 40], 'color': "#d1fae5"},
                            {'range': [40, 60], 'color': "#fff7ed"},
                            {'range': [60, 80], 'color': "#fff1f2"},
                            {'range': [80, 100], 'color': "#fee2e2"},
                        ],
                    }
                ))
                fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

            with right_col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                # textual recommendations
                if depression_prob < 0.2:
                    st.success(f"Very unlikely to have depression — {pct:.2f}%")
                    st.write("Tip: Maintain healthy habits, regular sleep and diet.")
                elif depression_prob < 0.4:
                    st.success(f"Unlikely to have depression — {pct:.2f}%")
                    st.write("Tip: Small lifestyle improvements and check-ins recommended.")
                elif depression_prob < 0.6:
                    st.warning(f"May have depression — {pct:.2f}%")
                    st.write("Recommend: Talk with a counselor; prioritize sleep & routine.")
                elif depression_prob < 0.8:
                    st.error(f"Likely to have depression — {pct:.2f}%")
                    st.write("Recommend: Seek professional help; contact campus counseling.")
                else:
                    st.error(f"Highly likely to have depression — {pct:.2f}%")
                    st.write("Immediate action: If suicidal thoughts are present, contact emergency services or helplines.")

                # Download input csv
                csv_buf = io.StringIO()
                input_df.to_csv(csv_buf, index=False)
                b64 = base64.b64encode(csv_buf.getvalue().encode()).decode()
                st.markdown(f"[⬇️ Download input CSV](data:file/csv;base64,{b64})")

                # Save last input/result to session
                st.session_state['last_input'] = input_df
                st.session_state['last_result'] = {'probability': depression_prob}
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.exception(e)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center' class='muted'>Made with ❤️ by ABHINAV CHATURVEDI — For educational purposes only. Not a substitute for professional diagnosis.</div>", unsafe_allow_html=True)
