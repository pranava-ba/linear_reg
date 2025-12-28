import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Linear Regression with Collinearity")

if st.button("Run Corrected Analysis"):
    np.random.seed(42)
    
    # Step 1: Generate x₁
    x_1 = np.random.randn(50)
    
    # Step 2: Create x₂ = 2 * x₁ (perfect collinearity)
    x_2 = 2 * x_1
    
    # Step 3: Generate y that ACTUALLY depends on x₁ and x₂
    true_beta1 = 2.5
    true_beta2 = 1.5
    true_intercept = 3.0
    
    y_1 = true_intercept + true_beta1*x_1 + true_beta2*x_2 + np.random.normal(0, 1, 50)
    
    # Step 4 & 5: Fit model
    X_model1 = np.column_stack([x_1, x_2])
    model1 = LinearRegression()
    model1.fit(X_model1, y_1)
    
    st.header("Model 1: y = β₀ + β₁·x₁ + β₂·x₂ (with x₂ = 2·x₁)")
    st.write(f"**True values:** Intercept = {true_intercept}, β₁ = {true_beta1}, β₂ = {true_beta2}")
    st.write(f"**Estimated:** Intercept = {model1.intercept_:.6f}")
    st.write(f"β₁ (x₁): {model1.coef_[0]:.6f}")
    st.write(f"β₂ (x₂): {model1.coef_[1]:.6f}")
    
    # Check the mathematical constraint
    st.write(f"\n**Note:** Since x₂ = 2·x₁, we need β₁ + 2β₂ = {true_beta1 + 2*true_beta2}")
    st.write(f"Estimated β₁ + 2β₂ = {model1.coef_[0] + 2*model1.coef_[1]:.6f}")
    
    st.divider()
    
    # Step 6: Create nearly collinear x₃
    x_3 = x_1 + 0.001 * np.random.uniform(0, 1, 50)
    
    # Step 7: Generate y₂ using the estimated coefficients from model1
    y_2 = model1.intercept_ + model1.coef_[0] * x_1 + model1.coef_[1] * x_2
    
    # Step 8: Fit with x₃ instead of x₂
    X_model2 = np.column_stack([x_1, x_3])
    model2 = LinearRegression()
    model2.fit(X_model2, y_2)
    
    st.header("Model 2: y = β₀ + β₁·x₁ + β₂·x₃")
    st.write(f"**True relationship is actually:** y = {model1.intercept_:.4f} + {model1.coef_[0]:.4f}·x₁ + {model1.coef_[1]:.4f}·x₂")
    st.write(f"**But x₃ ≈ x₁, not x₂!**")
    st.write(f"**Estimated:** Intercept = {model2.intercept_:.6f}")
    st.write(f"β₁ (x₁): {model2.coef_[0]:.6f}")
    st.write(f"β₂ (x₃): {model2.coef_[1]:.6f}")
    
    # Show the instability
    st.divider()
    st.header("Why This Shows Instability")
    st.write("""
    1. **Model 1 has perfect collinearity:** Infinite solutions exist
    2. **scikit-learn picks one arbitrary solution** (minimum norm)
    3. **Model 2 has near-collinearity:** x₃ ≈ x₁, so coefficients become unstable
    4. **The coefficients blow up** because small changes in x lead to huge changes in estimated βs
    """)
    
    # Compare predictions
    y_pred1 = model1.predict(X_model1)
    y_pred2 = model2.predict(X_model2)
    
    st.write(f"\n**Mean Squared Error (should be ~0 for both):**")
    st.write(f"Model 1 MSE: {np.mean((y_2 - y_pred1)**2):.10f}")
    st.write(f"Model 2 MSE: {np.mean((y_2 - y_pred2)**2):.10f}")
    
    st.divider()
    
    # Show small example
    st.header("Small Example (First 5 points)")
    data = {
        "x₁": x_1[:5],
        "x₂": x_2[:5],
        "x₃": x_3[:5],
        "y₁ (random)": y_1[:5],
        "y₂ (from model1)": y_2[:5]
    }
    st.table(data)

else:
    st.info("Click 'Run Corrected Analysis' to see the proper demonstration")
