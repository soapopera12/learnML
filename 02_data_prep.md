# 2 Data prep

## 2.1 Feature selection

### 2.1.1 Recursive Feature selection
### 2.1.2 Regression

## 2.2 Inliners

## 2.3 Outliers

An outlier is a data point that differs significantly from other observations.

### 2.3.1 Z-score

1. Compute the Mean (μ) of the dataset.
2. Compute the Standard Deviation (σ).
3. Calculate the Z-score for each data point using the formula:
   $$Z = \frac{X - \mu}{\sigma}$$
4. Set a Threshold for Outliers: |z| > 3. Data points with Z-scores exceeding the threshold are considered outliers.
5. X is the value of the current instance, μ is the mean and σ is the standard deviation of the feature.

### 2.3.2 IQR

1. Sort the Data in ascending order.
2. Find Q1 (25th percentile): Split the dataset in half and take the median of the lower half.
3. Find Q3 (75th percentile): Take the median of the upper half.
4. Compute IQR: Subtract Q1 from Q3.
5. Example:
   a) Sort the data: 3, 5, 7, 8, 12, 13, 14, 18, 21
   b) The lower half: 3, 5, 7, 8 → Median = (5 + 7) / 2 = 6
   c) The upper half: 13, 14, 18, 21 → Median = (14 + 18) / 2 = 16
   d) IQR = Q3 - Q1 = 16 - 6 = 10
   e) Detecting Outliers:
   f) Lower Bound = Q1 - 1.5 × IQR
   g) Upper Bound = Q3 + 1.5 × IQR
   h) Any value outside these bounds is considered an outlier.

## 2.4 Hypothesis testing

Hypothesis testing is a method used to make decisions or inferences about population parameters based on sample data.

Hypothesis testing begins with two competing hypotheses:
1. **Null hypothesis (H₀):** A statement of no effect or no difference. It often reflects a position of "status quo."
2. **Alternative hypothesis (H₁):** A statement that contradicts the null hypothesis. It suggests that there is an effect, a difference, or a relationship.

The goal of hypothesis testing is to determine whether the sample data provide enough evidence to reject the null hypothesis in favor of the alternative hypothesis.

**Types of Tests**
1. **One-tailed test:** Tests for the possibility of an effect in one direction.
2. **Two-tailed test:** Tests for the possibility of an effect in either direction.

**Key Statistical Values Used in Hypothesis Testing**
1. **P-value (Probability value):** If...
2. **Critical Value**
3. **Test Statistic**
4. **Significance Level (α)**
5. **Confidence Interval (CI):**

**Steps in Hypothesis Testing:**
1. State the hypotheses (null and alternative).
2. Select the significance level (α): Common choices are 0.05, 0.01, or 0.10.
3. Choose the appropriate test: z-test, t-test, chi-square test, etc.
4. Calculate the test statistic (z, t, or other).
5. Find the p-value associated with the test statistic.
6. Compare the p-value with α: If p ≤ α, reject H₀. If p > α, fail to reject H₀.
7. Make a conclusion based on the decision.

## 2.5 Imputation methods

1. Mean
2. Mode
3. Median
4. Constant
5. KNN
6. Multi-variate imputation by chained equation
7. Backward and forward fill
8. Removing row or column if needed

## 2.6 Encoding types

1. One hot encoding
2. Ordinal encoding
3. Label encoding
4. Positional encoding

## 2.7 Collinearity

### 2.7.1 Pearson Correlation (numerical features)

For variables X and Y with n observations:

$$r = \frac{\sum(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum(X_i - \bar{X})^2(Y_i - \bar{Y})^2}}$$

### 2.7.2 Chi-squared test (categorical features)

Builds a contingency table and computes the expected count. Calculate the Chi-squared statistic.

**Example:**

|     | Red | Blue |
|-----|-----|------|
| Male   | 10  | 20   |
| Female | 15  | 5    |

$$E_{ij} = \frac{(\text{row total}_i \times \text{col total}_j)}{\text{grand total}}$$

$$\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where Oij is the observed count and Eij is the expected count. Finally we check significance by comparing χ² to critical value from Chi-Squared distribution, or use a p-value