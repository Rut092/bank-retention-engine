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
st.divider()

# Create the three tabs
tab1, tab2, tab3 = st.tabs(["🎛️ Command Center", "📊 Raw Dataset", "🧠 Architecture & Intuition"])

with tab1:
    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("Demographic Information")
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.radio("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)

    with col2:
        st.subheader("Account Information")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=100.0)
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)

    with col3:
        st.subheader("Engagement Information")
        tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=3)
        num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
        subcol1,subcol2 = st.columns(2)
        with subcol1:
            has_cr_card = st.radio("Has Credit Card", ["Yes", "No"])
        with subcol2:
            is_active_member = st.radio("Is Active Member", ["Yes", "No"])


    st.divider()

    # Prediction Engine
    if st.button("Predict Retention Likelihood", type="primary", use_container_width=True):
        # Prepare input data

        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0
        gender_male = 1 if gender == "Male" else 0

        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [1 if has_cr_card == "Yes" else 0],
            'IsActiveMember': [1 if is_active_member == "Yes" else 0],
            'EstimatedSalary': [estimated_salary],
            'Geography_Germany': [geo_germany],
            'Geography_Spain': [geo_spain],
            'Gender_Male': [gender_male]
        })

        # 2. Execute the Pipeline Step-by-Step
        with st.spinner("Running through AI Pipeline..."):
            
            # Step A: Scale the raw data
            input_data_scaled = scaler.transform(input_data)
            
            # Step B: Compress using PCA
            input_data_pca = pca.transform(input_data_scaled)
            
            # Step C: Find Unsupervised Segment using K-Means
            cluster_label = kmeans.predict(input_data_pca)
            
            # Step D: Stack PCA components and Cluster Label for the SVM
            input_data_final = np.hstack((input_data_pca, cluster_label.reshape(-1, 1)))
            
            # 3. Make Prediction
            prediction = svm_model.predict(input_data_final)
            prediction_proba = svm_model.predict_proba(input_data_final)
            
            churn_prob = prediction_proba[0][1] * 100 
            
            # 4. Display Results
            st.subheader("Diagnostics")
            st.info(f"Unsupervised Segment: Cluster {cluster_label[0]}")
            
            if prediction[0] == 1:
                st.error(f"🚨 CRITICAL CHURN RISK: {churn_prob:.2f}%")
                st.markdown("This customer exhibits patterns matching historical exit data.")
            else:
                st.success(f"✅ SAFE (Low Risk): {churn_prob:.2f}% risk of exiting.")
                st.balloons()

with tab2:
    st.subheader("Raw Bank Churn Dataset")
    st.markdown("This is the 10,000-row historical dataset used to train the Hybrid Engine. Unique identifiers (RowNumber, CustomerId, Surname) were stripped to prevent data leakage prior to training.")
        
    try:
        df = pd.read_csv("dataset/Churn_Modelling.csv")
        st.dataframe(df, use_container_width=True)
    except FileNotFoundError:
        st.warning("Could not locate 'Churn_Modelling.csv'. Please ensure the dataset is in the exact same folder as app.py.")


with tab3:
    st.subheader("Architectural Overview & Engineering Intuition")
    st.markdown("A deep dive into the mathematics and engineering decisions behind the Hybrid Engine.")
    st.write("") 
    
    with st.container(border=True):
        st.markdown("### 🎯 The Problem: Beyond Basic Predictions")
        st.markdown("Predicting customer churn is easy. Predicting it accurately without overfitting is hard. Most basic pipelines just throw data at an algorithm and hope for the best. For this project, I wanted to build a mathematically rigorous **Hybrid Engine**—combining unsupervised pattern discovery with supervised precision classification.")

    with st.container(border=True):
        st.markdown("### 📉 Phase 1: Noise Reduction (PCA)")
        st.markdown("Bank datasets are highly dimensional. Before making any predictions, I applied **Principal Component Analysis (PCA)** to mathematically compress 11 categorical and numerical columns down to their most critical principal components, successfully retaining over 90% of the dataset's variance while stripping out the noise.")
        
        # Display the PCA graph (make sure pca_plot.png is in your folder)
        try:
            st.image("pca_plot.png", caption="Cumulative Explained Variance (90% Threshold)", use_container_width=True)
        except:
            st.info("🖼️ Save your PCA graph as 'pca_plot.png' in your project folder to display it here.")

    with st.container(border=True):
        st.markdown("### 🧩 Phase 2: Unsupervised Segmentation (K-Means)")
        st.markdown("Before telling the model *who* was going to leave, I wanted it to figure out *what types* of customers existed. I passed the PCA-reduced data into a **K-Means Clustering** algorithm. Instead of guessing the number of clusters, I optimized for the maximum **Silhouette Score**, which mathematically proved that the customer base naturally fractured into exactly **3 distinct segments**.")
        
        # Display the Silhouette graph
        try:
            st.image("silhouette_plot.png", caption="Silhouette Score Optimization Curve", use_container_width=True)
        except:
            st.info("🖼️ Save your Silhouette graph as 'silhouette_plot.png' in your project folder to display it here.")

    with st.container(border=True):
        st.markdown("### ⚙️ Phase 3 & 4: SVM Classification & Bayesian Optimization")
        st.markdown("For the final prediction, I engineered a new feature matrix combining the PCA components with the new K-Means cluster labels. I fed this into a **Support Vector Machine (SVM)**. I chose an SVM with an RBF kernel because human behavior is rarely linear; we needed an algorithm capable of wrapping complex, multi-dimensional decision boundaries around our customer segments.")
        st.markdown("Tuning an SVM via Grid Search is a brute-force approach. Instead, I deployed **Optuna** to run a Bayesian Optimization loop. Unlike Grid Search, Optuna learns from its past trials, probabilistically hunting down the absolute perfect balance between the $C$ (margin penalty) and $\gamma$ (kernel spread) hyperparameters.")

    with st.container(border=True):
        st.markdown("### 🏆 The Result: 86% Production Accuracy")
        st.markdown("The final Hybrid Pipeline achieved an **86% Accuracy** on unseen test data, proving that combining unsupervised pattern recognition with hyper-optimized supervised learning yields highly robust, enterprise-grade predictions.")
        
        # Split the bottom card into two columns for the final evaluation metrics
        colA, colB = st.columns(2)
        
        with colA:
            try:
                st.image("confusion_matrix.png", caption="Production Confusion Matrix", use_container_width=True)
            except:
                st.info("🖼️ Save your Confusion Matrix as 'confusion_matrix.png' to display it here.")
                
        with colB:
            try:
                st.image("roc_curve.png", caption="ROC-AUC Curve", use_container_width=True)
            except:
                st.info("🖼️ Save your ROC curve as 'roc_curve.png' to display it here.")