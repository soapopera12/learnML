# 10 Loss functions

## 10.1 Mean Squared Error (MSE)
MSE measures the average of squared differences between predicted and actual values. It penalizes larger errors more than MAE.
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Key characteristics**
1. Gives more weight to large errors (useful when large errors must be minimized).
2. Smooth and differentiable, making it easier for optimization in machine learning models.
3. Sensitive to outliers: A single large error increases MSE significantly.

## 10.2 Mean Absolute Error (MAE)
MAE measures the average absolute difference between predicted and actual values. It treats all errors equally.
$$MSE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
*(Note: The formula label in OCR was MSE but described MAE. MAE uses absolute values).*

1. Interpretable: Represents the average error in the same unit as the target variable.
2. Robust to outliers: Since it doesn’t square errors, it doesn’t over-penalize large errors.
3. Less sensitive to large errors (which may be a downside in some cases).

## 10.3 R squared
R-squared is a statistical measure that represents the goodness of fit of a regression model. It measures the variability in the dependent variable that is being explained by the independent variables.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Here $SS_{res}$ is the residual sum of squares and $SS_{tot}$ is the total sum of squares. The residual sum of squares is calculated by the summation of squares of perpendicular distance between data points and the best-fitted line. The total sum of squares is calculated by summation of squares of perpendicular distance between data points and the average line.

Adjusted R-Squared is an updated version of R-squared which takes account of the number of independent variables while calculating R-squared.

$$Adjusted\ R^2 = 1 - \frac{(1 - R^2) \cdot (n - 1)}{n - k - 1}$$

Here n is the number of observations, k is the number of independent variables.