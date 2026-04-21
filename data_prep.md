# Data prep

## Feature selection

### Feature Importance Methods

**Permutation Importance**

A model-agnostic method that measures feature importance by randomly shuffling each feature's values and observing how much the model's performance (e.g., accuracy or MSE) drops. The larger the drop, the more important the feature. It naturally captures interactions and is considered one of the most reliable ways to estimate true predictive importance.

1. Often considered most trustworthy
2. Computationally expensive
3. Sensitive to correlated features

**SHAP (SHapley Additive exPlanations)**

Based on game theory, SHAP assigns each feature a value representing its average marginal contribution across all possible coalitions. The mean absolute SHAP value provides robust global importance, while individual SHAP values provide local explanations.

1. Considered gold standard for interpretability
2. Computationally expensive (especially KernelSHAP)
3. TreeSHAP is fast but specific to tree-based models

**Tree-based Split Importance (Gain / Cover)**

For tree-based models (Random Forest, XGBoost, LightGBM), this built-in metric measures how much a feature improves impurity (gain) or how many samples it affects (cover) when used for splits.

1. Fast and convenient
2. Biased toward high-cardinality and numerical features
3. Can overstate importance of correlated features


**Drop-column Importance (LOCO)**

The most intuitive approach: train the model with all features, then retrain multiple times removing one feature at a time and measure the performance drop.

1. Conceptually very reliable
2. Extremely computationally expensive (requires retraining)
3. Does not handle correlated features well

**Standardized Coefficients (Linear Models)**

In linear or logistic regression (including regularized models like LASSO and Ridge), the magnitude of standardized coefficients reflects feature importance.

1. Simple and interpretable
2. Captures only linear relationships
3. Sensitive to feature scaling

**SHAP Interaction Values**

An extension of SHAP that quantifies both main effects and pairwise (or higher-order) feature interactions, providing a deeper understanding of how features work together.

1. Captures feature interactions explicitly
2. Computationally very expensive
3. Harder to summarize and interpret

**LIME (Local Interpretable Model-agnostic Explanations)**

LIME fits a simple interpretable model (usually linear) locally around a single prediction to approximate the behavior of a complex model.

1. Useful for local explanations
2. Model-agnostic
3. Less stable for global feature importance compared to SHAP or permutation


**Variance-based Methods (Sobol indices, functional ANOVA)**

These methods decompose the variance of the model output and attribute portions to individual features and their interactions.

1. Theoretically elegant
2. Focus on variance contribution rather than prediction error
3. Computationally intensive and less commonly used in practice

---

## Inliners

---

## outliers

An outlier is a data point that differs significantly from other observations.

---

### Z-score

1. Compute the Mean ($\mu$) of the dataset.
2. Compute the Standard Deviation ($\sigma$).
3. Calculate the Z-score for each data point using the formula.
$$Z = \frac{X - \mu}{\sigma}$$
4. Set a Threshold for Outliers: |z| > 3. Data points with Z-scores exceeding the threshold are considered outliers.
5. X is the values of the current instance, $\mu$ is the mean and $\sigma$ is the standard deviation of the feature    

---

### IQR

1. Sort the Data in ascending order.
2. Find Q1 (25th percentile): Split the dataset in half and take the median of the lower half.
3. Find Q3 (75th percentile): Take the median of the upper half.
4. Compute IQR: Subtract Q1 from Q3.
5. Example:
    1. Sort the data: 3, 5, 7, 8, 12, 13, 14, 18, 21
    2. The lower half: 3, 5, 7, 8 → Median = (5 + 7) / 2 = 6
    3. The upper half: 13, 14, 18, 21 → Median = (14 + 18) / 2 = 16
    4. $IQR = Q3−Q1 = 16−6 = 10$
    5. Detecting Outliers: 
    6. Lower Bound = $Q1−1.5×IQR$ 
    7. Upper Bound = $Q3+1.5×IQR$ 
    8. Any value outside these bounds is considered an outlier.

---

## Hypothesis testing

Hypothesis testing is a method used to make decisions or inferences about population parameters based on sample data.

Hypothesis testing begins with two competing hypotheses:

1. Null hypothesis ($H_o$): A statement of no effect or no difference. It often reflects a position of "status quo."
2. Alternative hypothesis ($H_1$): A statement that contradicts the null hypothesis. It suggests that there is an effect, a difference, or a relationship.

The goal of hypothesis testing is to determine whether the sample data provide enough evidence to reject the null hypothesis in favor of the alternative hypothesis.

Types of Tests

1. One-tailed test: Tests for the possibility of an effect in one direction (e.g., testing if a mean is greater than a certain value).
2. two-tailed test: Tests for the possibility of an effect in either direction (e.g., testing if a mean is different from a certain value).

Key Statistical Values Used in Hypothesis Testing

1. P-value (Probability value): If the p-value is low (typically < 0.05), it suggests that the observed result is unlikely under the null hypothesis, and you may reject H₀. If it is high, you fail to reject H₀.
2. Z-value (Z-score): The z-value is a measure of how many standard deviations an element is from the mean. It's used in z-tests, particularly when the sample size is large (typically > 30) or the population standard deviation is known. X is the sample mean, $\mu$ is the population mean, $\sigma$ is the population standard deviation.
$$
Z = \frac{X - \mu}{\sigma}
$$
2. T-value (T-score): The t-value is used in t-tests, typically when the sample size is small (n < 30) and the population standard deviation is unknown. S is the sample standard deviation, n is the sample size.
$$
t = \frac{X - \mu}{S / \sqrt{n}}
$$
3. Critical Value: The critical value is a threshold that defines the rejection region for the hypothesis test. For a given confidence level (e.g., 95\%), it’s the point where the probability of observing a value beyond it is equal to the significance level (α, often 0.05).
4. Confidence Interval (CI): A confidence interval provides a range of values within which the population parameter is likely to fall, with a certain level of confidence (e.g., 95\%).

Steps in Hypothesis Testing:

1. State the hypotheses (null and alternative).
2. Select the significance level ($\alpha$): Common choices are 0.05, 0.01, or 0.10.
3. Choose the appropriate test: z-test, t-test, chi-square test, etc.
4. Calculate the test statistic (z, t, or other).
5. Find the p-value associated with the test statistic.
6. Compare the p-value with $\alpha$: <br>
    If p $\leq$ $\alpha$, reject $H_o$.
    If p $>$ $\alpha$, fail to reject $H_o$.
7. Make a conclusion based on the decision

---

## Imputation methods

1. Mean
2. Mode
3. Median
4. Constant
5. knn 
6. Multi-variate imputation by chained equation
7. Backward and forward fill
8. removing row or column if needed

---

## Encoding Time to Angles Using Frequencies

Time values such as timestamps or time differences are not naturally meaningful to machine learning models because they are represented as scalar quantities with no inherent periodic structure. To make temporal information more expressive, time can be transformed into angular representations using multiple frequencies.

First, we define a set of frequencies that capture different temporal scales. These frequencies follow a geometric progression:

$$
\omega_j = \frac{1}{10^{j-1}}
$$

For example:

$$
\omega = [1, 0.1, 0.01, 0.001, \dots]
$$

Each frequency acts as a different \textit{zoom level} for observing time. Higher frequencies capture fine-grained temporal variations, while lower frequencies capture broader temporal trends.

Given two events, we compute the time difference:

$$
\Delta t = t_2 - t_1
$$

This time difference is multiplied with the frequency vector to produce a set of angular values:

$$
\theta_j = \omega_j \cdot \Delta t
$$

Thus, the time difference is projected across multiple temporal scales:

$$
\theta = [\omega_1 \Delta t, \omega_2 \Delta t, \omega_3 \Delta t, \dots]
$$

These angles are then mapped into a bounded range using trigonometric functions. A cosine transformation is commonly used:

$$
\text{Encoding}_j = \cos(\omega_j \Delta t)
$$

The cosine function maps all values into the interval:

$$
[-1, 1]
$$

This normalization ensures numerical stability and prevents very large time values from dominating the representation.

For example, if the time difference corresponds to one hour (e.g., $\Delta t = 3600$ seconds), multiplying by the frequency vector produces scaled angular values such as:

$$
[3600,; 360,; 36,; \dots]
$$

Applying the cosine transformation converts these angles into normalized values:

$$
[\cos(3600),; \cos(360),; \cos(36),; \dots]
$$

which lie in the range $[-1,1]$, for example:

$$
[-0.8,; -0.5,; 0.1,; \dots]
$$

Basically each of this value is trying to explain the model what is the content of this time in terms of a day, week, month, year etc... In summary, this encoding process converts raw time differences into a vector of periodic signals across multiple frequencies. Each frequency captures temporal patterns at a different scale, allowing machine learning models to interpret time relationships more effectively.

---

## Encoding Categorical features for tabular models


1. **One-Hot Encoding**:
Used for **nominal (unordered) categories**. Creates binary columns for each category to avoid introducing false order. \\
\textit{Example:} Color = \{Red, Blue, Green\}  
$$
\text{Red} \rightarrow [1,0,0], \quad
\text{Blue} \rightarrow [0,1,0], \quad
\text{Green} \rightarrow [0,0,1]
$$

1. **Ordinal Encoding**:  
Used for **ordered categories**. Maps categories to integers while preserving their natural order. \\
\textit{Example:} Size = \{Small < Medium < Large\}  
$$
\text{Small} \rightarrow 0, \quad
\text{Medium} \rightarrow 1, \quad
\text{Large} \rightarrow 2
$$

1. **Label Encoding**:
Assigns arbitrary integers to categories. Suitable for **tree-based models**, but may introduce misleading order for others. \\
\textit{Example:} Color = \{Red, Blue, Green\}  
$$
\text{Red} \rightarrow 0, \quad
\text{Blue} \rightarrow 1, \quad
\text{Green} \rightarrow 2
$$


