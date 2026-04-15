# Supervised Learning

## Linear regression

Regression in machine learning refers to a supervised learning technique where the goal is to predict a continuous numerical value based on one or more independent features.

1.  Objective: We find the hypothesis function which gives the target value.
Hypothesis function:
$$
y = \beta_0 + \beta_1 x + ... + \beta_n x + \epsilon
$$
2.  Optimize: we optimize this hypothesis function to reduce the loss/error. This can be done by manipulating the values of the coefficient using Normal equation, SVD, gradient descent, etc.
2.  Loss: Normal equation, SVD, gradient descent(MSE, MAE), etc.
3.  Assumes: Linear relationship, independence, Homoscedasticity, and normality of errors.

---

## Naive Bayes

1.  Supervised learning algorithm based on Bayes' Theorem
2.  Used to predict class $C_k$ given features $x = (x_1, x_2, \dots, x_n)$

**Bayes' Theorem:**

$$
P(C_k \mid x) = \frac{P(x \mid C_k)\, P(C_k)}{P(x)}
$$

1.  $P(C_k \mid x)$: Posterior (probability of class given features)
2.  $P(C_k)$: Prior (frequency of class) basically = $\frac{\text{no fo samples in spam}}{\text{Total number of samples}}$
3.  $P(x \mid C_k)$: Likelihood (probability of features given class)
4.  $P(x)$: Evidence (same for all classes, often ignored)


**Naive Assumption:**

1.  In the real world, features are often related. For example, if you are predicting "Rain," the features "Humidity" and "Cloud Cover" are highly correlated. The "Naive" Assumption: We assume that every feature is completely independent of every other feature, given the class. Mathematically, this simplifies the likelihood into a simple multiplication:
$$
P(x_1, x_2, \dots, x_n \mid C_k)
= \prod_{i=1}^{n} P(x_i \mid C_k)
$$
2.  This turns a complex multi-dimensional problem into a series of simple 1D problems.


**Training Process:**

1.  Compute priors:
$$
P(C_k) = \frac{\text{Number of samples in class } C_k}{\text{Total samples}}
$$
2.  Compute likelihoods:
$$
P(x_i \mid C_k) = \frac{\text{Frequency of } x_i \text{ in class } C_k}{\text{Total features in class } C_k}
$$
You do this for all the features and get their probabilities.
3.  Store probabilities for all features and classes   
4.  Compute score for each class
$$
\text{Score}(C_k) = P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)
$$
5.  Choose class with highest score


**Example:**

1.  Training dataset:

    Customer | City | Gender | Buy? |
    ---|---|---|---|
    |1 | Hyderabad | Male | Yes |
    2 | Mumbai | Female | Yes |
    3 | Hyderabad | Female | Yes |
    4 | Mumbai | Male | No |
    5 | Hyderabad | Male | No |

**Likelihood Calculation:**

1.  Class Yes (3 samples):

    $$
    P(\text{Hyderabad} \mid \text{Yes}) = \frac{2}{3}, \quad
    P(\text{Mumbai} \mid \text{Yes}) = \frac{1}{3}
    $$

    $$
    P(\text{Male} \mid \text{Yes}) = \frac{1}{3}, \quad
    P(\text{Female} \mid \text{Yes}) = \frac{2}{3}
    $$

2.  Class No (2 samples):

    $$
    P(\text{Hyderabad} \mid \text{No}) = \frac{1}{2}, \quad
    P(\text{Mumbai} \mid \text{No}) = \frac{1}{2}
    $$

    $$
    P(\text{Male} \mid \text{No}) = \frac{2}{2}, \quad
    P(\text{Female} \mid \text{No}) = \frac{0}{2}
    $$

**Prediction for New Data Point:**

1.  New customer: Male from Mumbai


$$
P(\text{Yes}) = \frac{3}{5}, \quad P(\text{No}) = \frac{2}{5}
$$

**Score for Yes:**

$$= P(Yes) \times P(Mumbai∣Yes) \times P(Male∣Yes)$$

$$
\frac{3}{5} \times \frac{1}{3} \times \frac{1}{3} = 0.066
$$

**Score for No:**

$$= P(No) \times P(Mumbai∣No) \times P(Male|No)$$

$$
\frac{2}{5} \times \frac{1}{2} \times \frac{2}{2} = 0.20
$$


2.  Since $0.20 > 0.066$, prediction is **No}

---

## Polynomial regression


## Classification


## Logistic regression


1. Objective: We find the hypothesis function which when passed through the sigmoid (logistic) activation function gives the target value.
Hypothesis function:
$$
\hat{y} = \frac{1}{e^{-(\beta_0 + \beta_1 x + ... + \beta_n x + \epsilon)}}
$$
2. Optimize: we optimize this activation output of the hypothesis function to reduce the loss/error. This is done by manipulating the values of the coefficient using gradient descent.
3. Loss: Log loss, maximum likelihood, etc

But since this is a classification problem it must give an output of 0 and 1. So how sigmoid function squeezes the input values between 0 and 1?

$$
Sigmoid(x) = \frac{1}{1 + e^{-x}}
$$

Value of e → euler’s constant → **2.7182818}

1.  x can be a positive or a negative number.
2.  If x is very large $e^{-x}$ becomes very small (close to 0), and 1 + 0 becomes 1 so the reciprocal of 1 is 1.
3.  If x is very small $e^{-x}$ becomes very large so the sum $1 + e^{-x}$ becomes very large and the reciprocal is close to 0.
4.  Therefore for a large value of x the function becomes 1 and for negative value the function becomes 0, the function output is 0.5 for value of x around 0.

In simple words:

1.  x $\rightarrow$ -ve $\rightarrow$ close to 0
2.  x $\rightarrow$ 0 $\rightarrow$ close to 0.5 
3.  x $\rightarrow$ +ve  $\rightarrow$ close to 1

---

## Multi-class or softmax regression

1. Objective: We find the hypothesis function which when passed through the sigmoid (logistic) activation function gives the target value.
Hypothesis function:
$$
\sigma(y)_k = \frac{e^{z_i}}{\sum_{j=1}^Ke^z_j}
$$ 
$$
\text{where} \ z_i =W_0^{k}x + W_1^{k}x + .... + W_n^{k}x + b
$$
2. Optimize: we optimize this activation output of the hypothesis function to reduce the loss/error. This is done by manipulating the values of the coefficient using gradient descent.
3. Loss: Log loss, maximum likelihood, etc


lets say there are k classes instead of 2. In this case we use softmax function instead of log loss.

**Lets understand with an example**

we use a categorical cross entropy loss function for this.

$$
LOSS = - log(P(y|X)) = -log(\frac{e^{z_y}}{\sum_{j=1}^Ke^z_j})
$$

1. Compute logits:

    $$
    z = Wx + b
    $$
    $$
    z = [z_0, z_1, z_2] \ \text{is computed for 3 classes}
    $$
2. Applying softmax:

    Lets say  
    $$
    P(y=0 ∣ x) = \frac{e^{z_{0}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}
    $$ 
    $$
    P(y=1 ∣ x)= \frac{e^{z_{1}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}
    $$
    $$
    P(y=2 ∣ x)= \frac{e^{z_{2}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}
    $$
3. Extract the Probability for the True Class

    $$
    \text{say values of} \ e^{z_{0}} = 2.0, e^{z_{1}} = 1.0  \ and \ e^{z_{2}} = 0.1
    $$

    $$ P(y=0 ∣ x) = \frac{e^{z_{0}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}} = \frac{e^{2.0}}{e^{2.0}+ e^{1.0}+ e^{0.1}} = 0.59$$

    $$ P(y=1 ∣ x) = \frac{e^{z_{1}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}} = \frac{e^{1.0}}{e^{2.0}+ e^{1.0}+ e^{0.1}} = 0.242$$

    $$ P(y=2 ∣ x)= \frac{e^{z_{2}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}} = \frac{e^{0.1}}{e^{2.0}+ e^{1.0}+ e^{0.1}} = 0.009$$

    $$\text{Remember} \ P(y=0 | x) + P(y=1 | x) + P(y=2 | x) = 1$$

4.  Compute the Cross-Entropy Loss for the Single Example

    This is only done for the correct label of y, here y should be 1

    $$ 
    \text{therefore} \ LOSS = -log(P(y = 1|x)) = -log(0.242) = -(-1.418) = 1.418
    $$

    $$
    \text{Total Loss} = \frac{1}{N} \sum_{i=1}^N -log(P(y^i | x^i))
    $$   

    So we only calculate loss for that target y value only

---

## Decision trees

Splitting criteria

1. Gini impurity: It is a measure of how impure (or mixed) a node is in terms of class distribution. 
    $$
    Gini = 1 - \sum p_i^2
    $$
    Example: Imagine a dataset where we classify loan approvals (Yes or No).
    Before splitting, we have 10 samples: 
    $$6 \ yes \rightarrow p_{yes} = \frac{6}{10} = 0.6$$
    $$4 \ no \rightarrow p_{no} = \frac{4}{10} = 0.4$$
    $$ Gini \ impurity = 1 - (0.6^2 + 0.4^2) = 0.48$$
    $$ \text{If a node is pure (only one class), Gini = 0.} $$

**Algorithms**

1.  CART
    1.  CART chooses the feature that results in the most homogeneous (pure) child nodes.
    2.  For classificaion: Uses Gini impurity (default) or entropy as a splitting criterion. where $p_i$ is the probability of class i in the node.
    For regression: Uses Mean Squared Error (MSE) or Mean Absolute Error (MAE) to find the best split
    $$
    Gini = 1 - \sum p_i^2
    $$
    $$
    MSE = \frac{1}{N}\sum (y_i - \hat{y})^2
    $$
    3.  The process continues until stopping conditions are met (e.g., max depth, min samples per leaf).
    4.  Each leaf node is assigned the most frequent class in that node.
1.  ID3
1.  CHAID




---

## K Nearest Neighbors (knn)

A standard KNN search (brute-force) compares a query point with every point in the dataset, resulting in $O(n)$ time per query. As the dataset grows, this becomes slow.

To speed this up, spatial data structures like KD-Trees and Ball Trees are used to skip large regions of space that cannot contain nearest neighbors.

---

### KD-Tree (K-Dimensional Tree)

A KD-Tree is a binary tree that partitions space using axis-aligned splits (hyperplanes).

**Core Idea**

Split space along coordinate axes (e.g., $x$, then $y$, then repeat (Both x and y are features) ).

**How it Builds**

1.  Choose a dimension (e.g., $x$-axis)
2.  Sort points along that axis
3.  Pick median point as split
4.  Divide data into left and right subtrees
5.  Recursively repeat, alternating dimensions

**Structure**

1.  Each node represents a point
2.  Space is divided into rectangular regions

**How KNN Search Works**

1.  Traverse down the tree to the leaf where query point belongs
2.  Keep track of current best distance
3.  Backtrack:
    1.  Check distance to splitting hyperplane
    2.  If this distance is smaller than best distance then explore other side.    
    3.  Else Prune entire subtree  

**Key Intuition**

1.  Eliminates large rectangular regions quickly
2.  Uses geometry of axis-aligned splits

**Limitation**

1.  Performance degrades in high dimensions (curse of dimensionality)
2.  Axis-aligned splits become inefficient


### Ball Tree

Ball Tree partitions space using hyperspheres (balls) instead of axis-aligned splits.

**Core Idea**

Group nearby points into clusters represented as balls (center + radius).

**How it Builds**

1.  Select two farthest points as pivots (pick a random point, now select the point farthest to it as A, then select another point Farthest to point A as point B) 
2.  Assign each point to nearest pivot → form two clusters
3.  For each cluster:
    1.  Compute center (mean of points)
    2.  Compute radius (max distance from center)
4.  Recursively split clusters

**Structure**

    1.  Each node represents a ball
    2.  Defined by:
        1.  Center $c$
        2.  Radius $r$

**How KNN Search Works**

1.  Traverse tree to reach closest ball
2.  Maintain current best distance
3.  For each node:
    1.  Compute distance to ball:
    $$
    d = ||q - c|| - r
    $$
    2.  If $d >$ current best distance:
        1.  Prune entire ball

**Key Intuition**

1.  Uses distance-based grouping instead of axis splits
2.  Can skip entire clusters efficiently
3.  Handles irregular data distributions better

---

### Locality Sensitive Hashing (LSH)

Locality Sensitive Hashing (LSH) is a technique used to efficiently find nearest neighbors by hashing data points such that **similar points are more likely to fall into the same bucket**.

**How it works**

Instead of comparing a query with all data points, LSH:

    1.  Applies hash functions to map points into buckets
    1.  Compares the query only with points in the same (or nearby) buckets
    1.  Uses multiple hash tables to improve recall and reduce misses

**Key idea**

Preserve similarity in hash space $\Rightarrow$ nearby points collide with high probability.

**Hash functions**
The choice of hash function depends on the similarity measure.

**For Euclidean distance ($L_2$)**
$$
h(x) = \left\lfloor \frac{a \cdot x + b}{w} \right\rfloor
$$

1.  $x \in \mathbb{R}^d$ : input **vector** (data point)
2.  $a \in \mathbb{R}^d$ : random **vector** (sampled from Gaussian distribution)
3.  $a \cdot x$ : **dot product** $\Rightarrow$ **scalar**
4.  $b \in \mathbb{R}$ : random **scalar** offset
5.  $w \in \mathbb{R}$ : **scalar** bucket width
6.  $h(x) \in \mathbb{Z}$ : **scalar integer** (bucket ID)

This method is called **random projection hashing**.

**For Cosine similarity (SimHash)**
$$
h(x) = \text{sign}(a \cdot x)
$$

1.  $x \in \mathbb{R}^d$ : input **vector**
2.  $a \in \mathbb{R}^d$ : random **vector**
3.  $a \cdot x$ : **scalar**
4.  $h(x) \in \{-1, +1\}$ : **scalar** hash value

**Intuition**
Data points are projected onto random directions, and the resulting scalar values are discretized into buckets. Similar points tend to produce similar projections and thus fall into the same buckets.

---

## Support Vector Machines (SVM)

A Support Vector Machine (SVM) is a supervised learning algorithm used for classification that finds a hyperplane separating data points with maximum margin.

**Problem Setup**

Given a dataset:
$$
(x_i, y_i), \quad x_i \in \mathbb{R}^d,\; y_i \in \{-1, +1\}
$$

A hyperplane is defined as:
$$
w^T x + b = 0
$$

Prediction:
$$
\hat{y} = \text{sign}(w^T x + b)
$$

**Margin Concept**

Two parallel hyperplanes are defined:
$$
w^T x + b = +1, \quad w^T x + b = -1
$$

The margin is the distance between these planes:
$$
\text{Margin} = \frac{2}{\|w\|}
$$

Maximizing margin is equivalent to minimizing:
$$
\|w\|^2
$$

**Margin Derivation**

**Distance from a Point to a Hyperplane**

Consider a hyperplane:
$$
w^T x + b = 0
$$

The perpendicular distance of a point $x_0$ from this hyperplane is given by:
$$
\text{Distance} = \frac{|w^T x_0 + b|}{\|w\|}
$$

**Margin Planes in SVM**

In SVM, we define two parallel hyperplanes:
$$
w^T x + b = +1 \quad \text{(positive class)}
$$
$$
w^T x + b = -1 \quad \text{(negative class)}
$$

These planes pass through the closest data points, known as **support vectors**.

**Distance Between the Two Planes**

Let $x_1$ be a point on the plane:
$$
w^T x_1 + b = 1
$$

The distance from this point to the other plane $w^T x + b = -1$ is:
$$
\text{Distance} =
\frac{|(w^T x_1 + b) - (-1)|}{\|w\|}
$$

This is a standard geometry formula for distance

Substituting:
$$
= \frac{|1 - (-1)|}{\|w\|}
= \frac{2}{\|w\|}
$$

**Final Result**

$$
\boxed{
\text{Margin} = \frac{2}{\|w\|}
}
$$

**Hard Margin SVM**

Optimization problem:
$$
\min_{w,b} \frac{1}{2} \|w\|^2
$$

Subject to:
$$
y_i (w^T x_i + b) \geq 1 \quad \forall i
$$

**Soft Margin SVM**

To handle non-separable data, introduce slack variables $\xi_i$:
$$
y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

New objective:
$$
\min_{w,b,\xi} \frac{1}{2} \|w\|^2 + C \sum_i \xi_i
$$

**Lagrangian Formulation (Intuition and SVM)**

**What is a Lagrangian?** 

A Lagrangian is a method to solve optimization problems with constraints.  
It converts a constrained problem into an unconstrained one by adding penalties for constraint violations.

**Simple Example** 

Maximize:
$$
f(x,y) = x + y
$$

Subject to:
$$
x + y = 1
$$

The Lagrangian is:
$$
\mathcal{L}(x,y,\lambda) = x + y + \lambda (1 - x - y)
$$

Here:

    1.  $x + y$ is the objective
    1.  $1 - x - y = 0$ is the constraint
    1.  $\lambda$ penalizes violations of the constraint


**General Form** 

For a problem:
$$
\min f(x) \quad \text{subject to} \quad g(x) \leq 0
$$

The Lagrangian is:
$$
\mathcal{L}(x,\lambda) = f(x) + \lambda g(x), \quad \lambda \geq 0
$$


**Soft-Margin SVM Formulation** 

Objective:
$$
\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_i \xi_i
$$

Constraints:
$$
y_i(w^T x_i + b) \geq 1 - \xi_i
$$
$$
\xi_i \geq 0
$$

**Rewrite Constraints**

$$
1 - \xi_i - y_i(w^T x_i + b) \leq 0
$$
$$
-\xi_i \leq 0
$$

**Lagrangian for Soft-Margin SVM** 

Introduce multipliers:
$$
\alpha_i \geq 0 \quad (\text{for margin constraints})
$$
$$
\mu_i \geq 0 \quad (\text{for } \xi_i \geq 0)
$$

The Lagrangian becomes:
$$
\mathcal{L} =
\frac{1}{2}\|w\|^2 + C \sum_i \xi_i
+ \sum_i \alpha_i \left(1 - \xi_i - y_i(w^T x_i + b)\right)
+ \sum_i \mu_i (-\xi_i)
$$

Simplified form:
$$
\mathcal{L} =
\frac{1}{2}\|w\|^2 + C \sum_i \xi_i
- \sum_i \alpha_i \left[y_i(w^T x_i + b) - 1 + \xi_i\right]
- \sum_i \mu_i \xi_i
$$

**Key Insight**

1.  Each constraint introduces a Lagrange multiplier
2.  $\alpha_i$ enforces correct classification
3.  $\mu_i$ ensures $\xi_i \geq 0$

**Dual Derivation via Substitution**

**Starting Lagrangian**

$$
\mathcal{L} =
\frac{1}{2} \|w\|^2 + C \sum_i \xi_i
- \sum_i \alpha_i \left(y_i(w^T x_i + b) - 1 + \xi_i \right)
- \sum_i \mu_i \xi_i
$$

**Step 1: Substitute $w$**

From optimality:
$$
w = \sum_i \alpha_i y_i x_i
$$

$$
\frac{1}{2} \|w\|^2 = \frac{1}{2} w^T w
$$

$$
= \frac{1}{2}
\left(\sum_i \alpha_i y_i x_i\right)^T
\left(\sum_j \alpha_j y_j x_j\right)
$$

$$
= \frac{1}{2}
\sum_i \sum_j
\alpha_i \alpha_j y_i y_j x_i^T x_j
$$

**Step 2: Substitute $w^T x_i$**

$$
w^T x_i =
\left(\sum_j \alpha_j y_j x_j\right)^T x_i
= \sum_j \alpha_j y_j x_j^T x_i
$$

**Step 3: Eliminate $b$**

$$
\sum_i \alpha_i y_i = 0
$$

All terms involving $b$ cancel out.

**Step 4: Eliminate $\xi_i$**

$$
C - \alpha_i - \mu_i = 0
\Rightarrow \mu_i = C - \alpha_i
$$

All $\xi_i$ terms cancel after substitution.

**Final Dual Form**

$$
\max_{\alpha}
\left(
\sum_i \alpha_i
- \frac{1}{2}
\sum_i \sum_j
\alpha_i \alpha_j y_i y_j x_i^T x_j
\right)
$$

Subject to:
$$
0 \leq \alpha_i \leq C
$$

$$
\sum_i \alpha_i y_i = 0
$$

**Dual Formulation**

Substituting optimal $w, b, \xi$ back into the Lagrangian, we obtain the dual:

$$
\max_{\alpha}
\sum_i \alpha_i
- \frac{1}{2}
\sum_i \sum_j
\alpha_i \alpha_j y_i y_j x_i^T x_j
$$

Subject to:
$$
0 \leq \alpha_i \leq C, \quad
\sum_i \alpha_i y_i = 0
$$

**Support Vectors**

Only points with:
$$
\alpha_i > 0
$$

contribute to the solution. These are called **support vectors}.

Weight vector:
$$
w = \sum_i \alpha_i y_i x_i
$$

**Training and Prediction in SVM**

**Step 1: Training Phase**

We solve the dual optimization problem:
$$
\max_{\alpha}
\left(
\sum_i \alpha_i
- \frac{1}{2}
\sum_i \sum_j
\alpha_i \alpha_j y_i y_j x_i^T x_j
\right)
$$

The solution gives optimal multipliers:
$$
\alpha_i^*
$$

**Step 2: Build the Model**

Using the optimal values, we compute:
$$
w = \sum_i \alpha_i y_i x_i
$$

**Step 3: Prediction Phase**

The decision function is:
$$
f(x) = w^T x + b
$$

Substitute $w$:
$$
f(x) =
\left(\sum_i \alpha_i y_i x_i\right)^T x + b
$$

$$
= \sum_i \alpha_i y_i (x_i^T x) + b
$$

**Decision Function**

The prediction for a new point $x$ is:
$$
f(x) = \sum_i \alpha_i y_i (x_i^T x) + b
$$

**Kernel Trick**

We replace dot products with kernels:
$$
K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
$$

Final decision function:
$$
f(x) = \sum_i \alpha_i y_i K(x_i, x) + b
$$

**Kernel Trick and Intuition**

**The Problem**

In basic SVM, we use the dot product:
$$
x_i^T x
$$

This operates in the original feature space and works well only when the data is linearly separable.

**The Idea**

If the data is not linearly separable, we map it to a higher-dimensional space:
$$
x \rightarrow \phi(x)
$$

In this new space, linear separation may become possible.

**Problem with Explicit Mapping**

Computing $\phi(x)$ explicitly can be:

1.  Computationally expensive
2.  Potentially infinite-dimensional (e.g., RBF kernel)

**Kernel Trick**

Instead of computing:
$$
\phi(x_i)^T \phi(x)
$$

we directly compute a kernel function:
$$
K(x_i, x)
$$

Thus, the kernel acts as a shortcut for the dot product in high-dimensional space.

**Intuition**

We behave as if we mapped data to a higher-dimensional space, without explicitly performing the transformation.

**Final Decision Function**

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x) + b
$$

**Interpretation**

For a new point $x$:

1.  Compare it with each training point $x_i$
2.  Measure similarity using $K(x_i, x)$
3.  Weight each contribution using:
    1.  $\alpha_i$ (importance)
    2.  $y_i$ (class label)
4.  Sum all contributions and add bias $b$

**Analogy**

1.  Each support vector acts like an influencer
2.  Kernel measures similarity to that influencer
3.  Final prediction is a weighted vote

**Example: RBF Kernel**

$$
K(x_i, x) = \exp\left(-\gamma \|x_i - x\|^2\right)
$$

1.  If $x$ is close to $x_i$, then $K(x_i, x) \approx 1$
2.  If $x$ is far from $x_i$, then $K(x_i, x) \approx 0$

Thus:

1.  Nearby points have higher influence
2.  Distant points contribute very little

**Key Insights**

1.  Maximizing margin improves generalization.
2.  Only support vectors determine the boundary.
3.  Dual formulation enables kernel methods.
4.  SVM optimization is convex, ensuring a global optimum.

---

## Artificial neural network

## Convolutional neural network

A Convolutional Neural Network (CNN) is a type of neural network designed to process grid-like data such as images. It leverages spatial structure and local patterns to efficiently learn representations.

**Translation invariancea** in CNNs refers to the ability of the network to recognize features regardless of their position in the image. This is achieved through convolutional layers, where the same filters (weights) are applied across different spatial locations, allowing the model to detect patterns anywhere. Pooling layers further enhance this by reducing sensitivity to small shifts, ensuring that slight translations in the input do not significantly change the output.

**Rotation invariance** in CNNs is not inherently built into standard architectures. Since convolutional filters are learned for specific orientations, a rotated version of an object may not be recognized unless the model has seen similar rotations during training. To achieve rotation invariance, techniques such as data augmentation (rotating images), or specialized architectures like rotation-equivariant networks are used.

**What it learns**
CNNs learn hierarchical features:

1.  Early layers: edges, corners
1.  Middle layers: textures, shapes
1.  Deeper layers: objects and high-level patterns

**Input Image Representation** 

An image is represented as a 3D tensor:
$$
H \times W \times C
$$
Example:
$$
224 \times 224 \times 3
$$
Here, $H$ = height, $W$ = width, $C$ = channels. RGB images have 3 channels (Red, Green, Blue), while other data may have different channels (e.g., grayscale = 1, feature maps in deeper layers = multiple channels).

**Convolution Layer**

The convolution layer applies filters (kernels) to extract features from the input.

**Filter / Kernel and Stride**

A filter is a small tensor of size:
$$
F \times F \times C
$$
Stride ($S$) determines how much the filter moves across the image.

**Filter with Multi-channel Input**

For an input with 3 channels, the filter also has depth 3. Each filter spans all input channels and produces a single feature map.

At each position:

1.  A local patch of size $F \times F \times C$ is taken from the input
2.  Each element of the patch is multiplied with a corresponding filter weight
3.  All values are summed and a bias is added


Mathematically:
$$
y = \sum_{i,j,k} x_{ijk} \cdot w_{ijk} + b
$$

Each $w_{ijk}$ is a learnable scalar weight corresponding to input value $x_{ijk}$.


**Small Example ($3 \times 3 \times 3$)**

Consider a small input patch:
$$
3 \times 3 \times 3
$$
and a filter of the same size:
$$
3 \times 3 \times 3
$$

Each channel of the filter contains weights. Each value in the input channels is multiplied with its corresponding weight:

1.  Channel 1 values $\times$ Channel 1 weights
2.  Channel 2 values $\times$ Channel 2 weights
3.  Channel 3 values $\times$ Channel 3 weights

All these products are summed:
$$
\sum_{i,j,k} x_{ijk} \cdot w_{ijk}
$$

This produces a single scalar output, representing how strongly the filter detects a pattern at that location.

**Padding and Output Size**

Padding ($P$) is used to control spatial size and preserve edge information.

Output size:
$$
\text{Output} = \frac{N - F + 2P}{S} + 1
$$

Where:

1.  $N$ = input size
2.  $F$ = filter size
3.  $P$ = padding
4.  $S$ = stride

Padding helps:

1.  Prevent shrinking of feature maps
2.  Preserve edge information

**Activation Function**

After convolution, a non-linear activation (ReLU) is applied:
$$
f(x) = \max(0, x)
$$

This introduces non-linearity, allowing the network to learn complex patterns.

**Flattening and Output**

After several convolution and pooling layers, feature maps are flattened into a vector:
$$
H \times W \times C \rightarrow \text{1D vector}
$$

This vector is passed to fully connected layers for final prediction.

---

## Recurrent neural network
## LSTM

## Deep neural network

## Graph neural network