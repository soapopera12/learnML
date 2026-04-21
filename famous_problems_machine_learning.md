# Famous problems in Machine learning

## Problems with Sigmoid

1. **Vanishing Gradient Problem**

    The Sigmoid function squashes inputs into a narrow range $(0,1)$:
    $$
    \sigma(x) = \frac{1}{1 + e^{-x}}
    $$

    Its derivative is:
    $$
    \sigma'(x) = \sigma(x)(1 - \sigma(x))
    $$

    The gradient is maximum at $x = 0$:
    $$
    \sigma'(0) = 0.25
    $$

    For large positive or negative inputs $(x > 5 \ \text{or} \ x < -5)$:
    $$
    \sigma'(x) \approx 0
    $$

    During backpropagation, gradients are multiplied across layers:
    $$
    \text{Update} \propto \prod_{k=1}^{n} \text{Gradient}_k
    $$

    If gradients are very small $(\approx 0)$:
    $$
    \text{Update} \approx 0
    $$

    This causes early layers to stop learning due to vanishing gradients.

2. **Not Zero-Centered**

    The output range of Sigmoid is:
    $$
    0 < \sigma(x) < 1
    $$

    Since outputs are always positive, gradients for weights tend to have the same sign:
    $$
    \frac{\partial L}{\partial w_i} \sim \text{same sign}
    $$

    This forces all weights to update in the same direction:
    $$
    w_i \uparrow \ \text{or} \ w_i \downarrow
    $$

    This leads to inefficient zig-zag optimization and slower convergence.

3. **Computational Expense**

    Sigmoid requires exponential computation:
    $$
    \sigma(x) = \frac{1}{1 + e^{-x}}
    $$

    This is more expensive compared to ReLU:
    $$
    \text{ReLU}(x) = \max(0, x)
    $$

In large-scale models, this increases computation time and resource usage.

---

## Zero Initialization Problem (Symmetry Issue)

When all weights in a layer are initialized to zero, every neuron produces the same output and receives the same gradient during backpropagation. As a result, all weights get updated identically, and neurons remain identical throughout training. The network effectively behaves as if each layer has only a single neuron. **This is called the symmetry problem.**

---

## Exploding Gradient Problem

When weights become too large in a deep network, it can lead to exploding gradients. In deep networks, activations are repeatedly multiplied by weights across layers. If weights are greater than 1 (e.g., around 2), the signal can grow exponentially (e.g., $2^{50}$), leading to very large values. They can be spotted by NaN loss, Wild fluctuations, Model instability. And they can be fixed by Gradient clipping, Batch normalization, Proper weight initialization etc.\\
During backpropagation, gradients are also multiplied layer by layer. If weights are large, gradients grow exponentially as they propagate backward. This results in extremely large updates:
$$
\text{Weight} = \text{Weight} - \text{Learning Rate} \times \text{Gradient}
$$
**This causes unstable training and divergence.**

---

## Vanishing Gradient Problem

If weights are too small (or certain activations like sigmoid are used), gradients shrink exponentially as they propagate backward. By the time they reach earlier layers, they become nearly zero. **This is the vanishing gradient problem, where early layers learn very slowly or not at all.**

---

## Why Linear activations do not work?

$$
y_1 = w_1 x + b_1, \quad 
y_2 = w_2 x + b_2, \quad 
y_3 = w_3 x + b_3
$$

Substituting $y_1$ in $y_2$

$$y_2 = (w_2 w_1)x + (w_2b_1 + b_2)$$

substituting this in $y_3$

$$
y_3 = (w_3 w_2 w_1)x + (w_3w_2b-1 + w_3b_2 + b_3)
$$

This proves even if more layers are added still the final equation is for linear form. Basically some weights are multiplied to again get the same linear form. No matter how many linear layers you stack, you only get another linear/affine map. 

---

## Why to avoid multi-collinearity

High multi-collinearity can lead to unstable estimates of coefficients and inflated standard errors and also provides duplicated information which is redundant. This is because the model cannot understand between two features with high multi-collinearity which one should be picked for.

---