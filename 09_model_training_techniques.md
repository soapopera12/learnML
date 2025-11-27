# 09_model_training_techniques.md
# 9 Model training techniques

## 9.1 Why Linear activations do not work?

$$y_1 = w_1 x + b_1, \quad y_2 = w_2 x + b_2, \quad y_3 = w_3 x + b_3$$

Substituting $y_1$ in $y_2$
$$y_2 = (w_2 w_1)x + (w_2 b_1 + b_2)$$

Substituting this in $y_3$
$$y_3 = (w_3 w_2 w_1)x + (w_3 w_2 b_1 + w_3 b_2 + b_3)$$

This proves even if more layers are added still the final equation is for linear form. Basically some weights are multiplied to again get the same linear form. No matter how many linear layers you stack, you only get another linear/affine map.

## 9.2 Avoiding multi-collinearity
High multi-collinearity can lead to unstable estimates of coefficients and inflated standard errors and also provides duplicated information which is redundant. This is because the model cannot understand between two features with high multi-collinearity which one should be picked for.

## 9.3 Exploding and vanishing gradients

## 9.4 Gradient descent

## 9.5 Hyper parameter tuning

## 9.6 Optimizers
An optimizer is an algorithm that updates the parameters (weights and biases) of a neural network to minimize the loss function.

There are two types of optimizers:
1. First-order optimization algorithms (uses only first-order derivatives). eg: SGD, RMSProp
2. Second-order optimization algorithms (uses only second-order derivatives). eg: Newton’s Method, L-BFGS.

1. **Momentum:** Helps accelerate SGD by adding a moving average of past gradients.
   $$v_t = \beta v_{t-1} + (1 - \beta)\nabla L(\theta)$$
   $$\theta = \theta - \eta v_t$$

1. **Momentum:** Helps accelerate SGD by adding a moving average of past gradients.
   $$v_t = \beta v_{t-1} + (1 - \beta)\nabla L(\theta)$$
   $$\theta = \theta - \eta v_t$$

2. **Nesterov Accelerated Gradient (NAG):** A modification of momentum that looks ahead before computing the gradient.
   $$v_t = \beta v_{t-1} + \eta \nabla L(\theta - \beta v_{t-1})$$
   $$\theta = \theta - v_t$$

3. **Adagrad (Adaptive Gradient Algorithm):** Adapts learning rate for each parameter based on past gradients.
   $$\theta = \theta - \frac{\eta}{\sqrt{G_t + \epsilon}} . \nabla L(\theta)$$

4. **Adam (Adaptive Moment Estimation):** Combines Momentum and RMSprop for adaptive learning rates with moving averages.
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla L(\theta)$$
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2)\nabla L(\theta)^2$$
   $$\theta = \theta - \frac{\eta}{\sqrt{v_t} + \epsilon} . m_t$$

## 9.7 Regularization
It solves the problem of overfitting, reduces model complexity and improves generalization.

### 9.7.1 L1 regularization (Lasso)
Penalty Term: The sum of the absolute values of the weights.
$$L1 = \lambda \sum |w_i|$$
Adds the sum of the absolute values of weights to the loss function. Encourages some weights to become exactly zero, effectively selecting only the most important features. Useful for sparse models and feature selection. The gradient of $|w|$ is either +1 or -1, creating a strong "pull" towards zero.

### 9.7.2 L2 regularization (Ridge)
$$L2 = \lambda \sum (w_i)^2$$
Adds the sum of the squared values of weights to the loss function. Shrinks weights towards zero but doesn’t make them exactly zero. Helps prevent overfitting while keeping all features. The gradient of $w^2$ is continuous and smooth (always 2w), so the model gradually decreases weights instead of forcing them to zero.

## 9.8 Transfer learning
## 9.9 Incremental learning
## 9.10 Memory optimization
1. Gradient accumulation
2. Freezing
3. Automatic mixed precision
4. 8-bit optimizers
5. Gradient checkpointing
6. Fast tokenizer
7. Dynamic padding
8. Uniform dynamic padding

## 9.11 Knowledge distillation

## Mixture of experts