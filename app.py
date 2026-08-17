import streamlit as st
import pandas as pd
import joblib

# Configure page
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff6b6b, #ee5a6f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .result-high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e72 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
    }
    .result-low-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved model, scaler, and expected columns
@st.cache_resource
def load_models():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_models()

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("❤️", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header-title">Heart Disease Risk Predictor</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("📋 **Please provide your health information to get a personalized heart disease risk assessment**")
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown("**Personal Information**")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("👤 Age (years)", 18, 100, 40)
with col2:
    sex = st.selectbox("👥 Sex", ["Male", "Female"], format_func=lambda x: "👨 Male" if x == "Male" else "👩 Female")
    sex = sex[0]  # Convert to M/F
with col3:
    st.info(f"**Age:** {age} years")

st.markdown("---")

# Health Metrics Section
st.markdown("**Blood Pressure & Cholesterol**")
col1, col2 = st.columns(2)

with col1:
    resting_bp = st.slider("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("🧬 Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    st.markdown("**Heart Activity**")
    max_hr = st.slider("💓 Max Heart Rate", 60, 220, 150)
    oldpeak = st.slider("📈 Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st.markdown("**Additional Metrics**")
col1, col2, col3 = st.columns(3)

with col1:
    chest_pain = st.selectbox("🫀 Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
with col2:
    fasting_bs = st.selectbox("🍽️ Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col3:
    exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["N", "Y"], format_func=lambda x: "No" if x == "N" else "Yes")

st.markdown("---")

# Advanced Section
st.markdown("**Advanced Settings**")
col1, col2 = st.columns(2)

with col1:
    resting_ecg = st.selectbox("📊 Resting ECG", ["Normal", "ST", "LVH"])
with col2:
    st_slope = st.selectbox("📉 ST Slope", ["Up", "Flat", "Down"])

st.markdown('</div>', unsafe_allow_html=True)

# Prediction section
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_button = st.button("🔍 Analyze Risk", use_container_width=True, type="primary")

# Display health summary before prediction
with st.expander("📋 Health Summary", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Age", f"{age} yrs")
    with col2:
        st.metric("Blood Pressure", f"{resting_bp} mm Hg")
    with col3:
        st.metric("Cholesterol", f"{cholesterol} mg/dL")
    with col4:
        st.metric("Max HR", f"{max_hr} bpm")

# Make prediction
if predict_button:
    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result with better styling
    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")
    
    if prediction == 1:
        st.markdown(
            '<div class="result-high-risk">⚠️ HIGH RISK of Heart Disease Detected</div>',
            unsafe_allow_html=True
        )
        st.warning("🚨 **Please consult a medical professional immediately** for proper evaluation and treatment.")
        
        with st.expander("💡 Recommendations for High Risk"):
            st.markdown("""
            - **Immediate Action:** Schedule an appointment with a cardiologist
            - **Lifestyle Changes:** 
              - Reduce sodium intake
              - Increase physical activity gradually
              - Manage stress through meditation or yoga
              - Avoid smoking and excessive alcohol
            - **Medical Monitoring:** Regular check-ups and ECG tests
            - **Diet:** Follow a heart-healthy diet (Mediterranean diet recommended)
            """)
    else:
        st.markdown(
            '<div class="result-low-risk">✅ LOW RISK - Your Heart Health Looks Good!</div>',
            unsafe_allow_html=True
        )
        st.success("💪 Keep maintaining a healthy lifestyle!")
        
        with st.expander("✨ Tips to Maintain Heart Health"):
            st.markdown("""
            - **Exercise:** Aim for 150 minutes of moderate activity per week
            - **Diet:** Eat plenty of fruits, vegetables, and whole grains
            - **Sleep:** Get 7-9 hours of quality sleep
            - **Stress Management:** Practice relaxation techniques
            - **Regular Check-ups:** Annual health screenings
            - **Avoid:** Smoking, excessive alcohol, and high-sodium foods
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; margin-top: 2rem;">
    <p>❤️ <strong>Disclaimer:</strong> This tool provides a risk assessment based on machine learning predictions. 
    It is NOT a substitute for professional medical advice. Always consult with a qualified healthcare provider.</p>
    <p><small>Created with ❤️ by Yash Darekar | Powered by Streamlit</small></p>
</div>
""", unsafe_allow_html=True)



#.venv\Scripts\streamlit run app.py
#.venv\Scripts\streamlit run app.py

#git init
#git add .
#git commit -m "Initial commit: Heart disease prediction app"
#git branch -M main
#git remote add origin https://github.com/Yashdarekar1/Heart_disease_app.git
#git push -u origin main