import streamlit as st
import numpy as np

st.title("Linear Regression with Multicollinearity")

# Manual Linear Regression Implementation
def linear_regression(X, y):
    # Add intercept term
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    # Calculate coefficients: beta = (X'X)^-1 X'y
    beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    return beta

if st.button("Run Analysis"):
    
    # Step 1: Generate x_1 (50 numeric numbers)
    np.random.seed(42)
    x_1 = np.random.randn(50)
    
    # Step 2: Find x_2 = 2 * x_1
    x_2 = 2 * x_1
    
    # Step 3: Generate response y ~ N(10, 25) where sigma^2 = 25
    y_1 = np.random.normal(10, np.sqrt(25), 50)
    
    # Step 4 & 5: Fit linear regression y = beta_1 * x_1 + beta_2 * x_2
    X_model1 = np.column_stack([x_1, x_2])
    beta_1 = linear_regression(X_model1, y_1)
    
    st.header("Model 1: y = β₁·x₁ + β₂·x₂")
    st.write(f"Intercept: {beta_1[0]:.4f}")
    st.write(f"β₁ (x₁): {beta_1[1]:.4f}")
    st.write(f"β₂ (x₂): {beta_1[2]:.4f}")
    
    st.divider()
    
    # Step 6: Create x_3 = x_1 + 0.001 * Uniform(0,1)
    x_3 = x_1 + 0.001 * np.random.uniform(0, 1, 50)
    
    # Step 7: Generate new response y ~ N(10, 25) where sigma^2 = 25
    y_2 = np.random.normal(10, np.sqrt(25), 50)
    
    # Step 8: Fit linear regression y = beta_1 * x_1 + beta_2 * x_2 + beta_3 * x_3
    X_model2 = np.column_stack([x_1, x_2, x_3])
    beta_2 = linear_regression(X_model2, y_2)
    
    st.header("Model 2: y = β₁·x₁ + β₂·x₂ + β₃·x₃")
    st.write(f"Intercept: {beta_2[0]:.4f}")
    st.write(f"β₁ (x₁): {beta_2[1]:.4f}")
    st.write(f"β₂ (x₂): {beta_2[2]:.4f}")
    st.write(f"β₃ (x₃): {beta_2[3]:.4f}")

else:
    st.info("Click 'Run Analysis' to generate data and fit regression models")
