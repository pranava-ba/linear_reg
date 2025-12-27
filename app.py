import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.title("Linear Regression ")
st.write("Generating a random vector")

# Generate data button
if st.button("Run Analysis"):
    
    # Step 1: Generate x_1 (50 numeric numbers)
    np.random.seed(42)
    x_1 = np.random.randn(50)
    
    # Step 2: Find x_2 = 2 * x_1
    x_2 = 2 * x_1
    
    # Step 3: Generate response y ~ N(10, 25)
    y_1 = np.random.normal(10, np.sqrt(25), 50)
    
    # Step 4 & 5: Fit linear regression y = beta_1 * x_1 + beta_2 * x_2
    X_model1 = np.column_stack([x_1, x_2])
    model1 = LinearRegression()
    model1.fit(X_model1, y_1)
    y_pred_1 = model1.predict(X_model1)
    r2_1 = model1.score(X_model1, y_1)
    
    st.header("Model 1: y = β₁·x₁ + β₂·x₂")
    st.write(f"**Coefficients:**")
    st.write(f"- β₁ (x₁): {model1.coef_[0]:.4f}")
    st.write(f"- β₂ (x₂): {model1.coef_[1]:.4f}")
    st.write(f"- Intercept: {model1.intercept_:.4f}")
    st.write(f"**R² Score:** {r2_1:.4f}")
    
    # Plot Model 1
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x_1, y=y_1, mode='markers', name='Actual y', marker=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=x_1, y=y_pred_1, mode='markers', name='Predicted y', marker=dict(color='red')))
    fig1.update_layout(title="Model 1: Actual vs Predicted", xaxis_title="x₁", yaxis_title="y")
    st.plotly_chart(fig1)
    
    st.divider()
    
    # Step 6: Create x_3 = x_1 + 0.001 * Uniform(0,1)
    x_3 = x_1 + (0.001 * np.random.uniform(0, 1, 50))
    
    # Step 7: Generate new response y ~ N(10, 25)
    y_2 = np.random.normal(10, np.sqrt(25), 50)
    
    # Step 8: Fit linear regression y = beta_1 * x_1 + beta_2 * x_2 + beta_3 * x_3
    X_model2 = np.column_stack([x_1, x_2, x_3])
    model2 = LinearRegression()
    model2.fit(X_model2, y_2)
    y_pred_2 = model2.predict(X_model2)
    r2_2 = model2.score(X_model2, y_2)
    
    st.header("Model 2: y = β₁·x₁ + β₂·x₂ + β₃·x₃")
    st.write(f"**Coefficients:**")
    st.write(f"- β₁ (x₁): {model2.coef_[0]:.4f}")
    st.write(f"- β₂ (x₂): {model2.coef_[1]:.4f}")
    st.write(f"- β₃ (x₃): {model2.coef_[2]:.4f}")
    st.write(f"- Intercept: {model2.intercept_:.4f}")
    st.write(f"**R² Score:** {r2_2:.4f}")
    
    # Plot Model 2
    '''fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_1, y=y_2, mode='markers', name='Actual y', marker=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=x_1, y=y_pred_2, mode='markers', name='Predicted y', marker=dict(color='red')))
    fig2.update_layout(title="Model 2: Actual vs Predicted", xaxis_title="x₁", yaxis_title="y")
    st.plotly_chart(fig2)
    
    st.divider()
    
    # Show correlation matrix
    st.header("Correlation Analysis")
    st.write("**Model 2 Predictor Correlations:**")
    corr_df = pd.DataFrame({'x₁': x_1, 'x₂': x_2, 'x₃': x_3})
    st.write(corr_df.corr())
    
    st.info("Notice: x₂ = 2·x₁ (perfect collinearity) and x₃ ≈ x₁ (near collinearity) cause unstable coefficient estimates")'''

else:
    st.info("Click 'Run Analysis' to generate data and fit regression models")
