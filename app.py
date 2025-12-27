import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Linear Regression ")

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
    model1 = LinearRegression()
    model1.fit(X_model1, y_1)
    
    st.header("Model 1: y = β₁·x₁ + β₂·x₂")
    st.write(f"Intercept: {model1.intercept_:.4f}")
    st.write(f"β₁ (x₁): {model1.coef_[0]:.4f}")
    st.write(f"β₂ (x₂): {model1.coef_[1]:.4f}")
    
    st.divider()
    
    # Step 6: Create x_3 = x_1 + 0.001 * Uniform(0,1)
    x_3 = x_1 + 0.001 * np.random.uniform(0, 1, 50)
    
    # Step 7: Generate new response y ~ N(10, 25) where sigma^2 = 25
    y_2 = np.random.normal(10, np.sqrt(25), 50)
    
    # Step 8: Fit linear regression y = beta_1 * x_1 + beta_2 * x_2 + beta_3 * x_3
    X_model2 = np.column_stack([x_1, x_2, x_3])
    model2 = LinearRegression()
    model2.fit(X_model2, y_2)
    
    st.header("Model 2: y = β₁·x₁ + β₂·x₂ + β₃·x₃")
    st.write(f"Intercept: {model2.intercept_:.4f}")
    st.write(f"β₁ (x₁): {model2.coef_[0]:.4f}")
    st.write(f"β₂ (x₂): {model2.coef_[1]:.4f}")
    st.write(f"β₃ (x₃): {model2.coef_[2]:.4f}")
    
    st.divider()
    
    # Show vectors
    st.header("Vector Values")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("x₁")
        st.write(x_1)
    
    with col2:
        st.subheader("x₂ = 2·x₁")
        st.write(x_2)
    
    with col3:
        st.subheader("x₃ = x₁ + 0.001·U(0,1)")
        st.write(x_3)

else:
    st.info("Click 'Run Analysis' to generate data and fit regression models")
