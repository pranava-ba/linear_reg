# Linear Regression

---

## Algorithm

- Generate a vector/column of 50 numeric numbers `x1`.
- Create another vector `x2 = 2 * x1`.
- Generate a response vector  `y1 ~ N(10, 25)`.
- Create `x3 = x1 + 0.001 * U(0, 1)` 
- Fit a linear regression model: `y1 ~ β1*x1 + β2*x2`.
- Fit a linear regression model: `y2 ~ β1*x1 + β2*x3`.
