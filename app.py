import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config
st.set_page_config(page_title="Bank Retention Engine", layout="wide")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    svm_model = joblib.load("bank_svm_model.pkl")
    scaler = joblib.load("bank_scaler.pkl")
    pca = joblib.load("bank_pca.pkl")
    kmeans = joblib.load("bank_kmeans.pkl")
    
    return svm_model, scaler, pca, kmeans


svm_model, scaler, pca, kmeans = load_model()

# App Header
st.title('🏦 Customer Retention Prediction Engine')
st.markdown("Enter customer details below to run them through Hybrid Supervised/Unsupervised Model AI Pipeline.")
