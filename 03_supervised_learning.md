# 3 Supervised Learning

## 3.1 Linear regression

Regression in machine learning refers to a supervised learning technique where the goal is to predict a continuous numerical value based on one or more independent features.

1. **Objective:** We find the hypothesis function which gives the target value. Hypothesis function:
   $$y = \beta_0 + \beta_1 x + ... + \beta_n x + \epsilon$$
2. **Optimize:** We optimize this hypothesis function to reduce the loss/error. This can be done by manipulating the values of the coefficient using Normal equation, SVD, gradient descent, etc.
3. **Loss:** Normal equation, SVD, gradient descent (MSE, MAE), etc.
4. **Assumes:** Linear relationship, independence, Homoscedasticity, and normality of errors.

## 3.2 Polynomial regression

## 3.3 Classification

## 3.4 Logistic regression

1. **Objective:** We find the hypothesis function which when passed through the sigmoid (logistic) activation function gives the target value. Hypothesis function:
   $$\hat{y} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x + \dots + \beta_n x)}}$$
2. **Optimize:** We optimize this activation output of the hypothesis function to reduce the loss/error. This is done by manipulating the values of the coefficient using gradient descent.
3. **Loss:** Log loss, maximum likelihood, etc.

But since this is a classification problem it must give an output of 0 and 1.
So how does the sigmoid function squeeze the input values between 0 and 1?

$$Sigmoid(x) = \frac{1}{1 + e^{-x}}$$

Value of e → Euler's constant → 2.7182818

1. x can be a positive or a negative number.
2. If x is very large, e^{-x} becomes very small (close to 0), and 1 + 0 becomes 1 so the reciprocal of 1 is 1.
3. If x is very small, e^{-x} becomes very large so the sum 1 + e^{-x} becomes very large and the reciprocal is close to 0.
4. Therefore for a large value of x the function becomes 1 and for negative value ...

... (rest of your original content continues unchanged)

## 3.6 Decision trees

**Splitting criteria**
1. **Gini impurity:** It is a measure of how impure (or mixed) a node is in terms of class distribution.
   $$Gini = 1 - \sum p_i^2$$

## 3.6 Decision trees

**Splitting criteria**
1. **Gini impurity:** It is a measure of how impure (or mixed) a node is in terms of class distribution.
   $$Gini = 1 - \sum p_i^2$$
   **Example:** Imagine a dataset where we classify loan approvals (Yes or No). Before splitting, we have 10 samples:
   6 yes $\rightarrow p_{yes} = \frac{6}{10} = 0.6$
   4 no $\rightarrow p_{no} = \frac{4}{10} = 0.4$
   $Gini\ impurity = 1 - (0.6^2 + 0.4^2) = 0.48$
   If a node is pure (only one class), Gini = 0.

**Algorithms**
1. **CART**
   a) CART chooses the feature that results in the most homogeneous (pure) child nodes.
   b) For classification: Uses Gini impurity (default) or entropy as a splitting criterion.
      $$Gini = 1 - \sum p_i^2$$
      where $p_i$ is the probability of class i in the node.
      For regression: Uses Mean Squared Error (MSE) or Mean Absolute Error (MAE) to find the best split.
      $$MSE = \frac{1}{N} \sum (y_i - \hat{y})^2$$
   c) The process continues until stopping conditions are met (e.g., max depth, min samples per leaf).
   d) Each leaf node is assigned the most frequent class in that node.
2. **ID3**
3. **CHAID**

## 3.7 Artificial neural network
## 3.8 Convolutional neural network
## 3.9 Recurrent neural network
## 3.10 Deep neural network
## 3.11 Graph neural network