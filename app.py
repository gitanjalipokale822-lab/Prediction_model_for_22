import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Prediction App", page_icon="🤖")

# Custom CSS for styling and animations
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-out;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# UI Layout
st.title("🤖 ML Predictor")
st.write("Enter the details to get a prediction.")

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Inputs
    gender_input = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    salary = st.number_input("Estimated Salary", min_value=15000, max_value=200000, value=50000, step=1000)

    # Preprocessing
    # Note: Assuming 0 for Male, 1 for Female (Verify this matches your training data)
    gender_val = 0 if gender_input == "Male" else 1
    
    # Prediction
    if st.button("Predict Now"):
        # Creating DataFrame in the exact order the model expects
        input_data = pd.DataFrame([[gender_val, age, salary]], 
                                  columns=['Gender', 'Age', 'EstimatedSalary'])
        
        prediction = model.predict(input_data)
        
        # Displaying result
        st.success(f"Prediction Result: {prediction[0]}")
        st.balloons()
        
    st.markdown('</div>', unsafe_allow_html=True)
