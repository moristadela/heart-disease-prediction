import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Heart Disease Risk Checker Application",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Load model components
@st.cache_resource
def load_model():
    """Load the trained model components"""
    try:
        model = joblib.load("heart_disease_model.pkl")
        encoding_maps = None
        feature_names = joblib.load("feature_names.pkl")
        return {
            'model': model,
            'encoding_maps': encoding_maps,
            'feature_names': feature_names
        }
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.stop()

def predict_heart_disease(data, model_components):
    """Make heart risk checker using the trained model"""
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Get components
    model = model_components['model']
    encoding_maps = model_components['encoding_maps']
    feature_names = model_components['feature_names']
    
    # Apply encodings to categorical columns
    # for column in df.columns:
     #   if column in encoding_maps and column != 'income':
     #       df[column] = df[column].map(encoding_maps[column])
    
    # Ensure we only use features that the model was trained on
    df_for_pred = df[feature_names].copy()
    
    # Make prediction
    prediction = model.predict(df_for_pred)[0]
    probabilities = model.predict_proba(df_for_pred)[0]
    
    # Get income label
    prediction_label = 'Ada Penyakit' if prediction == 1 else 'Tidak Ada Penyakit'
    
    return {
        'prediction': int(prediction),
        'prediction_label': prediction_label,
        'probability': float(probabilities[prediction]),
        'probabilities': probabilities.tolist()
    }

def validate_inputs(data):
    """Validate input data"""
    errors = []
    
    if data['age'] < 17 or data['age'] > 100:
        errors.append("Umur harus antara 17 dan 100 tahun")
    
    if data['trestbps'] < 80 or data['trestbps'] > 200:
        errors.append("Tekanan darah harus antara 80 dan 200 mm Hg")

    if data['chol'] < 100 or data['chol'] > 600:
        errors.append("Kolesterol total harus antara 100 dan 600 mg/dl")

    if data['thalach'] < 70 or data['thalach'] > 220:
        errors.append("Denyut jantung maksimum harus antara 70 dan 220 bpm")

    if data['oldpeak'] < 0 or data['oldpeak'] > 10:
        errors.append("Nilai oldpeak harus antara 0.0 dan 10.0")

    return errors

def export_prediction(data, result):
    """Export prediction result to JSON"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'input_data': data,
        'prediction': {
            'class': result['prediction_label'],
            'confidence': result['probability'],
            'raw_prediction': result['prediction']
        }
    }
    return json.dumps(export_data, indent=2)

def reset_session_state():
    """Reset all input values to default"""
    keys_to_reset = [
       'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# Load model
model_components = load_model()

# Define mappings (from the original notebook)
# Opsi untuk jenis kelamin
sex_options = ['Laki-laki', 'Perempuan']

# Opsi untuk tipe nyeri dada (cp)
cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']

# Opsi untuk kadar gula darah puasa (fbs)
fbs_options = ['≤ 120 mg/dl', '> 120 mg/dl']

# Opsi untuk hasil EKG saat istirahat (restecg)
restecg_options = [0, 1, 2]  # atau bisa diganti jadi ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy']

# Opsi untuk angina yang dipicu oleh olahraga (exang)
exang_options = ['Ya', 'Tidak']

# Opsi untuk slope (kemiringan segmen ST)
slope_options = [1, 2, 3]

# Opsi untuk jumlah pembuluh darah besar (ca)
ca_options = [0, 1, 2, 3]

# Opsi untuk thal (hasil tes thalium)
thal_options = [3, 6, 7]  # 3: normal, 6: fixed defect, 7: reversible defect

# Opsi untuk target (hasil prediksi, untuk label/visualisasi)
target_options = ['Ada Penyakit', 'Tidak Ada Penyakit']


# Main app
st.title("🫀 Heart Disease Risk Checker Application ")
st.markdown("Prediksi risiko penyakit jantung berdasarkan data klinis dan demografis pasien.")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Features")
    
    # Create form for inputs
    # Create form for inputs
with st.form("prediction_form"):
    st.markdown("### 📝 Input Data Pasien")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Umur", min_value=17, max_value=100, value=50, key="age")
        sex = st.selectbox("Jenis Kelamin", sex_options, key="sex")
        cp = st.selectbox("Tipe Nyeri Dada (Chest Pain Type)", cp_options, key="cp")
        trestbps = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=80, max_value=200, value=120, key="trestbps")
        chol = st.number_input("Kolesterol Total (mg/dl)", min_value=100, max_value=600, value=240, key="chol")
        fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", fbs_options, key="fbs")
        restecg = st.selectbox("Hasil EKG Saat Istirahat", restecg_options, key="restecg")

    with col2:
        thalach = st.number_input("Denyut Jantung Maksimum (bpm)", min_value=70, max_value=220, value=150, key="thalach")
        exang = st.selectbox("Nyeri Dada Saat Olahraga?", exang_options, key="exang")
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="oldpeak")
        slope = st.selectbox("Slope Segmen ST", slope_options, key="slope")
        ca = st.selectbox("Jumlah Pembuluh Darah Besar yang Terlihat (ca)", ca_options, key="ca")
        thal = st.selectbox("Hasil Tes Thalium", thal_options, key="thal")

    st.divider()

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        predict_button = st.form_submit_button("🔮 Prediksi", type="primary")
    with col_btn2:
        reset_button = st.form_submit_button("🔄 Reset")
    with col_btn3:
        export_button = st.form_submit_button("📤 Export Hasil")


# Handle reset button
if reset_button:
    reset_session_state()
    st.rerun()

# Handle prediction
if predict_button:
    # Collect input data
    input_data = {
      'age': age,
        'sex': 1 if sex == 'Laki-laki' else 0,
        'cp': {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}[cp],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 0 if fbs == '≤ 120 mg/dl' else 1,
        'restecg': int(restecg),
        'thalach': thalach,
        'exang': 1 if exang == 'Ya' else 0,
        'oldpeak': oldpeak,
        'slope': int(slope),  # slope sudah numerik (1, 2, 3)
        'ca': int(ca),        # ca sudah numerik (0–3)
        'thal': int(thal)     # thal sudah numerik (3, 6, 7)
    }
    
    # Validate inputs
    validation_errors = validate_inputs(input_data)
    
    if validation_errors:
        with col2:
            st.error("❌ Validation Errors:")
            for error in validation_errors:
                st.error(f"• {error}")
    else:
        # Make prediction
        try:
            result = predict_heart_disease(input_data, model_components)
            
            # Store result in session state for export
            st.session_state['last_prediction'] = {
                'input_data': input_data,
                'result': result
            }
            
            with col2:
                st.subheader("🎯 Prediction Results")
                
                # Display prediction
                prediction_color = "green" if result['prediction_label'] == '>50K' else "orange"
                st.markdown(f"**Prediksi Risiko Penyakit Jantung:** :{prediction_color}[{result['prediction_label']}]")
                
                # Confidence level with gauge
                confidence = result['probability'] * 100
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Level (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': prediction_color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Probability breakdown
                prob_df = pd.DataFrame({
                    'Class': ['Tidak Ada Penyakit', 'Ada Penyakit'],
                    'Probability': result['probabilities']
                })
                
                fig_bar = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    title='Probability Distribution',
                    color='Probability',
                    color_continuous_scale=['orange', 'green']
                )
                fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)
                
        except Exception as e:
            with col2:
                st.error(f"❌ Prediction Error: {str(e)}")

# Feature Importance section
st.subheader("📊 Feature Importance")

if 'model' in model_components:
    try:
        feature_names = model_components['feature_names']
        feature_importance = model_components['model'].feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Feature Importance in Decision Tree Model',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_importance, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

# Handle export
if export_button:
    if 'last_prediction' in st.session_state:
        export_data = export_prediction(
            st.session_state['last_prediction']['input_data'],
            st.session_state['last_prediction']['result']
        )
        
        st.download_button(
            label="📥 Download Prediction Results",
            data=export_data,
            file_name=f"predict_heart_disease()_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ No prediction results to export. Please make a prediction first.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit Rendyta Moristadella A12.2022.06769• *")