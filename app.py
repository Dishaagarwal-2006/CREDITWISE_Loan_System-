"""
Credit Loan Approval Prediction System
A machine learning-powered app to predict loan approval using 3 classification models
Built with Streamlit | Trained on 1000+ loan applications
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from preprocessing import LoanDataPreprocessor, get_input_constraints, get_feature_descriptions

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .approved-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .rejected-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    """Load all trained models and preprocessing artifacts"""
    try:
        log_model = joblib.load('models/logistic_regression_model.pkl')
        nb_model = joblib.load('models/naive_bayes_model.pkl')
        knn_model = joblib.load('models/knn_model.pkl')
        preprocessor = LoanDataPreprocessor()
        return log_model, nb_model, knn_model, preprocessor
    except FileNotFoundError:
        st.error("❌ Models not found! Please ensure models are saved in the 'models' directory.")
        st.stop()

log_model, nb_model, knn_model, preprocessor = load_models()

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.markdown("# 📋 Navigation")
page = st.sidebar.radio("Select Section", 
    ["🏠 Home", "🔮 Predict Loan", "📊 Model Performance", "ℹ️ About"])

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<h1 class="header-title">💰 Loan Approval Predictor</h1>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### 🎯 What This App Does:
        - **Predicts** whether your loan will be approved
        - **Compares** 3 ML models (Logistic Regression, Naive Bayes, KNN)
        - **Explains** feature importance and decision factors
        - **Tracks** model performance metrics
        """)
    
    with col2:
        st.success("""
        ### 🚀 Key Features:
        - **Best Model**: Naive Bayes (87% accuracy)
        - **Dataset**: 1,000+ loan applications
        - **Features**: 18 input variables
        - **Real-time**: Instant predictions
        """)
    
    st.markdown("---")
    
    # Statistics Cards
    st.subheader("📈 Model Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Logistic Regression ⭐</h3>
            <h2>87.5%</h2>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>Naive Bayes </h3>
            <h2>87%</h2>
            <p>Accuracy (BEST)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>KNN</h3>
            <h2>74%</h2>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature importance explanation
    st.subheader("🎲 Key Factors for Loan Approval")
    
    factors = {
        "Credit Score": "Higher score = Better approval chances",
        "Income": "Stable income improves approval odds",
        "Existing Loans": "Fewer existing loans help approval",
        "DTI Ratio": "Lower debt-to-income is favorable",
        "Savings": "Strong savings increase approval rate"
    }
    
    for factor, description in factors.items():
        st.write(f"✓ **{factor}**: {description}")

# ==================== PREDICTION PAGE ====================
elif page == "🔮 Predict Loan":
    st.markdown('<h1 class="header-title">🔮 Predict Loan Approval</h1>', 
                unsafe_allow_html=True)
    
    st.write("Enter your loan application details below to get an instant prediction!")
    
    # Create input form in columns
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.subheader("📊 Personal Information")
        
        input_data['Age'] = st.slider(
            "Age (years)", 
            min_value=18, max_value=65, value=35,
            help="Your age in years"
        )
        
        input_data['Gender'] = st.selectbox(
            "Gender",
            options=['Male', 'Female']
        )
        
        input_data['Marital_Status'] = st.selectbox(
            "Marital Status",
            options=['Single', 'Married', 'Divorced', 'Widowed']
        )
        
        input_data['Dependents'] = st.slider(
            "Number of Dependents",
            min_value=0, max_value=10, value=0
        )
        
        input_data['Education_Level'] = st.selectbox(
            "Education Level",
            options=['Not Graduate', 'Graduate', 'Postgraduate']
        )
    
    with col2:
        st.subheader("💼 Income & Employment")
        
        input_data['Applicant_Income'] = st.number_input(
            "Your Monthly Income (₹)",
            min_value=0, max_value=500000, value=50000, step=1000
        )
        
        input_data['Coapplicant_Income'] = st.number_input(
            "Co-applicant Monthly Income (₹)",
            min_value=0, max_value=500000, value=0, step=1000
        )
        
        input_data['Employment_Status'] = st.selectbox(
            "Employment Status",
            options=['Salaried', 'Self-employed', 'Unemployed']
        )
        
        input_data['Employer_Category'] = st.selectbox(
            "Employer Category",
            options=['Private', 'Government', 'Self-employed']
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💳 Credit & Financial")
        
        input_data['Credit_Score'] = st.slider(
            "Credit Score",
            min_value=300, max_value=900, value=650,
            help="Higher credit score improves approval chances"
        )
        
        input_data['Existing_Loans'] = st.slider(
            "Number of Existing Loans",
            min_value=0, max_value=10, value=0
        )
        
        input_data['Savings'] = st.number_input(
            "Total Savings (₹)",
            min_value=0, max_value=10000000, value=100000, step=10000
        )
        
        input_data['DTI_Ratio'] = st.slider(
            "Debt-to-Income Ratio",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            help="Ratio of existing debt to income. Lower is better."
        )
    
    with col2:
        st.subheader("🏠 Loan & Property Details")
        
        input_data['Loan_Amount'] = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=10000, max_value=10000000, value=500000, step=10000
        )
        
        input_data['Loan_Term'] = st.slider(
            "Loan Duration (months)",
            min_value=6, max_value=360, value=180, step=6
        )
        
        input_data['Loan_Purpose'] = st.selectbox(
            "Loan Purpose",
            options=['Personal', 'Car', 'Business', 'Home', 'Education']
        )
        
        input_data['Property_Area'] = st.selectbox(
            "Property Area",
            options=['Urban', 'Semiurban', 'Rural']
        )
        
        input_data['Collateral_Value'] = st.number_input(
            "Collateral Value (₹)",
            min_value=0, max_value=10000000, value=500000, step=10000
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🎯 Get Prediction", use_container_width=True, type="primary"):
        try:
            # Preprocess input
            features = preprocessor.preprocess_input(input_data)
            
            # Get predictions from all models
            pred_log = log_model.predict(features.reshape(1, -1))[0]
            pred_nb = nb_model.predict(features.reshape(1, -1))[0]
            pred_knn = knn_model.predict(features.reshape(1, -1))[0]
            
            # Get probabilities
            prob_log = log_model.predict_proba(features.reshape(1, -1))[0][1]
            prob_nb = nb_model.predict_proba(features.reshape(1, -1))[0][1]
            prob_knn = knn_model.predict_proba(features.reshape(1, -1))[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            models = [
                ("Logistic Regression", pred_log, prob_log, 0.875),
                ("Naive Bayes ⭐", pred_nb, prob_nb, 0.87),
                ("KNN", pred_knn, prob_knn, 0.74)
            ]
            
            cols = [col1, col2, col3]
            
            for i, (model_name, prediction, probability, accuracy) in enumerate(models):
                with cols[i]:
                    approval = "✅ APPROVED" if prediction == 1 else "❌ REJECTED"
                    color = "green" if prediction == 1 else "red"
                    
                    st.metric(
                        label=model_name,
                        value=approval,
                        delta=f"{probability:.1%} confidence",
                        delta_color="off"
                    )
                    
                    st.write(f"Model Accuracy: **{accuracy:.1%}**")
            
            st.markdown("---")
            
            # Consensus prediction
            avg_probability = (prob_log + prob_nb + prob_knn) / 3
            consensus = "✅ LIKELY APPROVED" if avg_probability > 0.5 else "❌ LIKELY REJECTED"
            
            if avg_probability > 0.5:
                st.markdown(f"""
                <div class="approved-card">
                    <h3>Consensus: {consensus}</h3>
                    <p>Average Approval Probability: <strong>{avg_probability:.1%}</strong></p>
                    <p>This prediction is based on the agreement of multiple ML models</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rejected-card">
                    <h3>Consensus: {consensus}</h3>
                    <p>Average Approval Probability: <strong>{avg_probability:.1%}</strong></p>
                    <p>This prediction is based on the agreement of multiple ML models</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization
            st.markdown("---")
            st.subheader("📊 Model Confidence Comparison")
            
            fig = go.Figure(data=[
                go.Bar(
                    name='Approval Probability',
                    x=['Logistic\nRegression', 'Naive Bayes', 'KNN'],
                    y=[prob_log, prob_nb, prob_knn],
                    marker_color=['#667eea', '#f5576c', '#764ba2'],
                    text=[f'{p:.1%}' for p in [prob_log, prob_nb, prob_knn]],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Model Confidence for Loan Approval",
                xaxis_title="ML Model",
                yaxis_title="Approval Probability",
                hovermode='x unified',
                height=400,
                showlegend=False,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check your input values and try again.")

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "📊 Model Performance":
    st.markdown('<h1 class="header-title">📊 Model Performance Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Performance metrics
    st.subheader("📈 Detailed Model Metrics")
    
    metrics_data = {
        'Model': ['Logistic Regression', 'Naive Bayes', 'KNN'],
        'Accuracy': [0.875, 0.87, 0.74],
        'Precision': [0.790, 0.797, 0.596],
        'Recall': [0.803, 0.770, 0.459],
        'F1 Score': [0.797, 0.783, 0.519]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.bar(df_metrics, x='Model', y='Accuracy', 
                     title='Accuracy Comparison', 
                     color='Accuracy',
                     color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcf7f'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df_metrics, x='Model', y='Precision',
                     title='Precision Comparison',
                     color='Precision',
                     color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcf7f'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.bar(df_metrics, x='Model', y='Recall',
                     title='Recall Comparison',
                     color='Recall',
                     color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcf7f'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Full Metrics Table")
    st.dataframe(df_metrics.set_index('Model'), use_container_width=True)
    
    st.markdown("---")
    st.subheader("🏆 Model Rankings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### Best for Different Metrics:
        
        - **Overall Accuracy**: Logistic Regression (87.5%)
        - **Precision** (Fewer False Positives): Naive Bayes (79.7%)
        - **Recall** (Fewer False Negatives): Logistic Regression (80.3%)
        """)
    
    with col2:
        st.success("""
        ### Why Naive Bayes Wins:
        
        After feature engineering:
        - Improved from 84.5% → 87% accuracy
        - Better generalization
        - Consistent across all metrics
        - Recommended as primary model ⭐
        """)

# ==================== ABOUT PAGE ====================
elif page == "ℹ️ About":
    st.markdown('<h1 class="header-title">ℹ️ About This Project</h1>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.write("""
        This is a **Loan Approval Prediction System** built with Machine Learning.
        
        **Dataset**: 1,000+ historical loan applications
        
        **Objective**: Predict whether a loan application will be approved
        
        **Models Used**:
        - Logistic Regression
        - Naive Bayes (Best Performer)
        - K-Nearest Neighbors
        """)
    
    with col2:
        st.subheader("💻 Technology Stack")
        st.write("""
        **Backend**: Python 3.8+
        - scikit-learn (Machine Learning)
        - pandas (Data Processing)
        - numpy (Numerical Computing)
        
        **Frontend**: Streamlit
        
        **Visualization**: Plotly
        
        **Deployment**: Streamlit Cloud
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Dataset Features (18 Input Variables)")
    
    features_df = pd.DataFrame({
        'Feature': [
            'Age', 'Gender', 'Marital Status', 'Dependents', 'Education',
            'Applicant Income', 'Co-applicant Income', 'Employment Status',
            'Credit Score', 'Existing Loans', 'DTI Ratio', 'Savings',
            'Loan Amount', 'Loan Term', 'Loan Purpose', 'Property Area',
            'Collateral Value', 'Employer Category'
        ],
        'Type': ['Numeric', 'Categorical', 'Categorical', 'Numeric', 'Categorical',
                'Numeric', 'Numeric', 'Categorical', 'Numeric', 'Numeric',
                'Numeric', 'Numeric', 'Numeric', 'Numeric', 'Categorical',
                'Categorical', 'Numeric', 'Categorical']
    })
    
    st.dataframe(features_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔬 Model Development Process")
    
    st.write("""
    **Phase 1: Data Preprocessing**
    - Handled missing values
    - Encoded categorical variables
    - Scaled numerical features using StandardScaler
    
    **Phase 2: Initial Model Training**
    - Logistic Regression: 87.5% ✓
    - Naive Bayes: 84.5%
    - KNN: 76.5%
    
    **Phase 3: Feature Engineering**
    - Created new meaningful features
    - Removed redundant features
    - Optimized feature set
    
    **Phase 4: Final Model Tuning**
    - Naive Bayes: 87% (Improved!) ⭐
    - Selected as primary model
    """)
    
    st.markdown("---")
    
    st.subheader("👨‍💻 Developer")
    st.write("""
    Built as a **Portfolio Project** for demonstrating:
    - ML model development and evaluation
    - Data preprocessing and feature engineering
    - Web application development with Streamlit
    - Model deployment and productionization
    """)
    
    st.markdown("---")
    
    st.subheader("📝 Disclaimer")
    st.warning("""
    **Important**: This model is for educational purposes. 
    Real loan approval decisions should involve:
    - Human judgment and domain expertise
    - Additional verification and background checks
    - Legal and compliance review
    - Assessment of applicant credibility
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💰 Credit Loan Approval Predictor | Built with Streamlit & ML</p>
    <p>© 2026 | Portfolio Project</p>
</div>
""", unsafe_allow_html=True)