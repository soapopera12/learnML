# Unsupervised Learning

## Clustering

### k-means clustering

---

### k-means++ clustering

---

### GMM (Gaussian Mixture Model)

Instead of assigning points to a single cluster, GMM computes the probability that each point belongs to each cluster (soft assignment). GMMs are trained using the Expectation-Maximization (EM) algorithm:

1. In the E-step, the model calculates the probability (responsibility) of each data point belonging to each Gaussian component.
2. In the M-step, it updates the parameters of each Gaussian (mean, covariance, and mixing coefficient) to maximize the likelihood of the data.

GMMs are the probabilistic versions of k-means clustering. K-means forces each point to be a part of a hard circle. However GMMs on the other hand gives each point a probability eg:- 70\% probability x belongs to cluster a or 30\% it belongs to cluster b.

For each cluster (bell curve / normal) in the mixture.

1. Mean ($\mu$).
2. Variance ($\sigma^2$) for single feature or $\Sigma$ for multivariate.
3. Weight ($\pi$) Total data points belonging in this cluster.

GMM does not rely on simple distance measures; it uses an iterative optimization process called Expectation-Maximization (EM). It models data as a mixture of $K$ Gaussian distributions. Here is the PDF and normal equation for any normal equation.

**Probability Density Function:**

$$
P(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

1. $\pi_k$: Mixing weight of cluster $k$
2. $\mu_k$: Mean of cluster $k$
3. $\Sigma_k$: Covariance matrix of cluster $k$
4. $\mathcal{N}$: Gaussian distribution

**Gaussian Distribution:**

$$
\mathcal{N}(x \mid \mu, \Sigma) =
\frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}
\exp \left( -\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu) \right)
$$

1. $|\Sigma|$: Determinant of covariance matrix
2. $\Sigma^{-1}$: Inverse of covariance matrix
3. $(x - \mu)$: Distance from mean
4. $d$: Number of features

**The following the EM algorithm used by GMMs**

**Step A: Initialization**

1. Initialize $\mu_k$, $\Sigma_k$, and $\pi_k$ randomly

**Step B: Expectation (E-Step)**

1. Compute responsibility $\gamma_{nk}$ for each data point $x_n$

$$
\gamma_{nk} =
\frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}
{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}
$$

1. Represents probability that cluster $k$ generated $x_n$

**Step C: Maximization (M-Step)**

1. Update parameters using responsibilities

$$
N_k = \sum_{n=1}^{N} \gamma_{nk}
$$

**Updated Mean:**

$$
\mu_k = \frac{1}{N_k} \sum_{n=1}^{N} \gamma_{nk} x_n
$$

**Updated Covariance:**

$$
\Sigma_k =
\frac{1}{N_k}
\sum_{n=1}^{N}
\gamma_{nk} (x_n - \mu_k)(x_n - \mu_k)^T
$$

**Updated Mixing Coefficient:**

$$
\pi_k = \frac{N_k}{N}
$$

**Repeat Until Convergence**

1. Alternate between E-step and M-step
2. Stop when parameters change very little

---

### DBScan

---

### Agglomerative Hierarchical Clustering

---

## Dimensionality reduction

It is used to avoid multi-collinearity, reduce high dimensionality, remove noise and have better generalization.

---

### Curse of dimensionality

PCA is a dimensionality reduction technique that transforms your high-dimensional data into a smaller number of uncorrelated variables called principal components, while preserving as much variance (information) as possible.

---

### PCA

1. **Center the data:**

    Subtract the mean from each feature so that the dataset is centered around zero.  
    This ensures PCA focuses only on variance, not absolute values.

2. **Understand how the data varies:**

    Compute the covariance matrix:
    $$
    C = \frac{1}{m} X^\top X
    $$
    This matrix tells us:
    1. Variance of each feature (diagonal entries)
    2. How features move together (off-diagonal entries)

3. **Find the main directions (principal components):** 

    Compute eigenvectors and eigenvalues of C:
    $$
    C v = \lambda v
    $$
    1. Eigenvectors = directions in which data spreads out
    2. Eigenvalues = how much spread (importance) in that direction

    Pick the top k eigenvectors with the largest eigenvalues.

4. **Project the data:**

    Form a matrix W using the top k eigenvectors and project:
    $$
    X_{\text{new}} = X W
    $$
    Now the data is represented in fewer dimensions while keeping most of the important variation.

**Key Idea:**

PCA rotates the data to a new coordinate system where:

1. The first axis captures the most variance
2. The second captures the next most, and so on

Then it keeps only the top k directions and discards the rest.

Explanation of $C v = \lambda $

1. **Solving the eigenvalue equation:**

    We start with:
    $$
    C v = \lambda v
    $$
    Rewrite it as:
    $$
    (C - \lambda I)v = 0
    $$
    For a non-zero vector v, this has a solution only if:
    $$
    \det(C - \lambda I) = 0
    $$
    This is called the **characteristic equation**.

2. **Example (2** $\times$ **2 case):**

    Let:
    $$
    C =
    \begin{bmatrix}
    2 & 1 \\
    1 & 2
    \end{bmatrix}
    $$

3. **Step 1: Solve for** $\lambda$ **:**

    $$
    \det(C - \lambda I) =
    \begin{vmatrix}
    2 - \lambda & 1 \\
    1 & 2 - \lambda
    \end{vmatrix}
    = (2 - \lambda)^2 - 1 = 0
    $$
    $$
    (2 - \lambda)^2 = 1 \quad \Rightarrow \quad \lambda = 3, 1
    $$

4. **Step 2: Solve for v:**

    For $\lambda$ = 3:
    $$
    (C - 3I)v =
    \begin{bmatrix}
    -1 & 1 \\
    1 & -1
    \end{bmatrix} v = 0
    $$
    This gives:
    $$
    v =
    \begin{bmatrix}
    1 \\
    1
    \end{bmatrix}
    $$

    For $\lambda$ = 1:
    $$
    (C - I)v =
    \begin{bmatrix}
    1 & 1 \\
    1 & 1
    \end{bmatrix} v = 0
    $$
    This gives:
    $$
    v =
    \begin{bmatrix}
    1 \\
    -1
    \end{bmatrix}
    $$

5. **Interpretation**

1. Direction $\begin{bmatrix}1 \\ 1\end{bmatrix}$ → eigenvalue $3$ → high variance  
2. Direction $\begin{bmatrix}1 \\ -1\end{bmatrix}$ → eigenvalue $1$ → low variance  

6. **Connection to PCA**

1. Keep direction $\begin{bmatrix}1 \\ 1\end{bmatrix}$  
2. Drop direction $\begin{bmatrix}1 \\ -1\end{bmatrix}$ when reducing to 1D

Also this is important to know but correlated features are combined into principal components, where one direction captures most of the shared variance (large eigenvalue), and remaining directions capture little variance. So PCA is immune to correlated features. If two features are highly correlated → they vary together. PCA captures this with one principal component (direction of maximum variance)

1. Standardization: Scale data to have zero mean and unit variance.
2. Relationship identification: Calculate covariance matrix.
3. Maximizing Variance: calculate Eigenvectors/Eigenvalues.
4. Feature transformation: using top k eigenvectors to transform feature matrix. 

---

### t-SNE


1. **Convert distances to probabilities (high-dimensional space):**

    For data points $x_1, \dots, x_n$, define conditional probabilities using a Gaussian kernel:
    $$
    p_{j|i} = \frac{\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right)}{\sum_{k \neq i} \exp\left(-\frac{\|x_i - x_k\|^2}{2\sigma_i^2}\right)}
    $$
    Then symmetrize to obtain joint probabilities:
    $$
    p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
    $$
    This is final probabilities that we have calculated with actual data.

2. **Define similarities in low-dimensional space:**

    Map points to low-dimensional representations $y_1, \dots, y_n$ (random low dim representation of original points), and define:
    $$
    q_{ij} = \frac{\left(1 + \|y_i - y_j\|^2\right)^{-1}}{\sum_{k \neq l} \left(1 + \|y_k - y_l\|^2\right)^{-1}}
    $$
    This corresponds to a Student t-distribution with one degree of freedom. This is the projection onto a 2D or 3D space that we are trying to learn/plot.

3. **Define the objective function (KL divergence):**

    The goal is to match the two distributions by minimizing:
    $$
    \mathcal{L} = KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
    $$
    This loss is basically like how much information is lost if we approximate P using Q.

4. **Compute the gradient:**  

    Taking derivative with respect to \(y_i\):
    $$
    \frac{\partial \mathcal{L}}{\partial y_i}
    =
    4 \sum_j (p_{ij} - q_{ij}) (y_i - y_j)\left(1 + \|y_i - y_j\|^2\right)^{-1}
    $$

5. **Optimize embeddings:**

    Initialize $y_i$ randomly and update using gradient descent:
    $$
    y_i \leftarrow y_i - \eta \frac{\partial \mathcal{L}}{\partial y_i}
    $$
    Optionally apply momentum and early exaggeration to improve convergence.

**Why the t-distribution?**

Using a Gaussian in low dimensions leads to the \textit{crowding problem}, where many moderately distant points collapse together. The Student t-distribution has heavier tails:
$$
(1 + \|y_i - y_j\|^2)^{-1}
$$
which allows dissimilar points to remain far apart, improving separation in the embedding.

---

### LDA

---

## Hidden Markov Models (HMMs)


## Variational Auto Encoders (VAE)

## Generative Adversarial Networks (GAN)

GANs consist of two neural networks: a Generator and a Discriminator, trained simultaneously in a adversarial setup. The Generator creates fake data (e.g., images) from random noise, while the Discriminator tries to distinguish between real data (from the true dataset) and the fake data produced by the Generator. The magic happens because they improve together.


## Diffusion Models