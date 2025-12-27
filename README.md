# Linear Regression

---

## Algorithm

- Generate a vector/column of 50 numeric numbers `x1`.
- Create another vector `x2 = 2 * x1`.
- Generate a response vector `y1` from a normal distribution with mean = 10 and variance = 25 (i.e., `y1 ~ N(10, 25)`).
- Create `x3 = x1 + 0.001 * U(0, 1)`, where `U(0, 1)` is a random value from a uniform distribution between 0 and 1.
- Fit a linear regression model: `y1 ~ β1*x1 + β2*x2`.
- Fit a linear regression model: `y1 ~ β1*x1 + β2*x2 + β3*x3`.
