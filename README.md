# Linear Regression

---

## Algorithm

- Generate a vector/column of 50 numeric numbers \( x_1 \).
- Find another vector \( x_2 = 2x_1 \).
- Generate a response vector \( y_1 \sim \mathcal{N}(10, 25) \) (variance \( \sigma^2 = 25 \)).
- Create \( x_3 = x_1 + 0.001 \cdot \text{Uniform}(0, 1) \).
- Fit a linear regression model: \( y_1 = \beta_1 x_1 + \beta_2 x_2 + \varepsilon \).
- Fit a linear regression model: \( y_1 = \beta_1 x_1 + \beta_2 x_3 +  \varepsilon \).
