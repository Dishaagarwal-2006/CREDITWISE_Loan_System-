"""
preprocessing.py - Preprocessing pipeline for credit loan approval prediction
Handles data cleaning, encoding, and scaling
"""

import pandas as pd
import numpy as np
import joblib
import os

class LoanDataPreprocessor:
    """
    Preprocesses loan application data for model prediction
    """
    
    def __init__(self, scaler_path='models/scaler.pkl', 
                 encoders_path='models/label_encoders.pkl',
                 features_path='models/feature_names.pkl'):
        """
        Load preprocessing artifacts
        """
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        self.feature_names = joblib.load(features_path)
    
    def preprocess_input(self, input_dict):
        """
        Convert user input dictionary to scaled features ready for prediction
        
        Parameters:
        -----------
        input_dict : dict
            Dictionary with user inputs
            
        Returns:
        --------
        np.array : Scaled features ready for model prediction
        """
        
        # Create dataframe from input
        df = pd.DataFrame([input_dict])
        
        # Encode categorical variables
        df_encoded = df.copy()
        
        for col in self.label_encoders.keys():
            if col in df_encoded.columns:
                try:
                    # Use transform for known values
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
                except ValueError:
                    # If unknown value, use first class
                    df_encoded[col] = self.label_encoders[col].transform([
                        self.label_encoders[col].classes_[0]
                    ])[0]
        
        # Select features in correct order
        X = df_encoded[self.feature_names]
        
        # Scale numerical features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled[0]  # Return single sample


def get_feature_descriptions():
    """
    Return descriptions of all input features for the UI
    """
    descriptions = {
        'Applicant_Income': 'Monthly income of applicant (₹)',
        'Coapplicant_Income': 'Monthly income of co-applicant (₹)',
        'Employment_Status': 'Employment type (Salaried/Self-employed/Unemployed)',
        'Age': 'Age of applicant (years)',
        'Marital_Status': 'Marital status',
        'Dependents': 'Number of dependents',
        'Credit_Score': 'Credit score (300-900)',
        'Existing_Loans': 'Number of existing loans',
        'DTI_Ratio': 'Debt-to-Income ratio (0-1)',
        'Savings': 'Savings amount (₹)',
        'Collateral_Value': 'Collateral value (₹)',
        'Loan_Amount': 'Requested loan amount (₹)',
        'Loan_Term': 'Loan duration (months)',
        'Loan_Purpose': 'Purpose of loan',
        'Property_Area': 'Type of property area',
        'Education_Level': 'Education qualification',
        'Gender': 'Gender',
        'Employer_Category': 'Employer type'
    }
    return descriptions


def get_input_constraints():
    """
    Return input constraints for validation and UI hints
    """
    constraints = {
        'Applicant_Income': {'min': 0, 'max': 200000, 'type': 'numeric'},
        'Coapplicant_Income': {'min': 0, 'max': 200000, 'type': 'numeric'},
        'Employment_Status': {'options': ['Salaried', 'Self-employed', 'Unemployed']},
        'Age': {'min': 18, 'max': 65, 'type': 'numeric'},
        'Marital_Status': {'options': ['Single', 'Married', 'Divorced', 'Widowed']},
        'Dependents': {'min': 0, 'max': 10, 'type': 'numeric'},
        'Credit_Score': {'min': 300, 'max': 900, 'type': 'numeric'},
        'Existing_Loans': {'min': 0, 'max': 10, 'type': 'numeric'},
        'DTI_Ratio': {'min': 0, 'max': 1, 'type': 'numeric'},
        'Savings': {'min': 0, 'max': 5000000, 'type': 'numeric'},
        'Collateral_Value': {'min': 0, 'max': 5000000, 'type': 'numeric'},
        'Loan_Amount': {'min': 1000, 'max': 5000000, 'type': 'numeric'},
        'Loan_Term': {'min': 6, 'max': 360, 'type': 'numeric'},
        'Loan_Purpose': {'options': ['Personal', 'Car', 'Business', 'Home', 'Education']},
        'Property_Area': {'options': ['Urban', 'Semiurban', 'Rural']},
        'Education_Level': {'options': ['Not Graduate', 'Graduate', 'Postgraduate']},
        'Gender': {'options': ['Male', 'Female']},
        'Employer_Category': {'options': ['Private', 'Government', 'Self-employed']}
    }
    return constraints