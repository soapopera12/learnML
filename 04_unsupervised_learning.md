# 4 Unsupervised Learning

## 4.1 Clustering

## 4.2 Dimensionality reduction
It is used to avoid multi-collinearity, reduce high dimensionality, remove noise and have better generalization.

### 4.2.1 Curse of dimensionality
PCA is a dimensionality reduction technique that transforms your high-dimensional data into a smaller number of uncorrelated variables called principal components, while preserving as much variance (information) as possible.

### 4.2.2 PCA
1. **Centering:** First, PCA centers the data by subtracting the mean of each feature, so that the dataset has zero mean.
2. **Covariance Matrix:** Computes the covariance matrix of the data, which captures how the features vary and co-vary with each other.
   $$C = \begin{bmatrix} Var(X_1) & Covar(X_1, X_2) \\ Covar(X_2, X_1) & Var(X_2) \end{bmatrix}$$
3. **Computing eigen vectors and eigenvalues:** Eigen vectors are special vectors that do not change direction on applying a matrix transformation. They reveal the direction of max variance (principal components) of the co-variance matrix. Eigen values determine the stretch or the variance in each direction of eigen vectors. More stretch is equal to more variance. The eigen vectors are sorted and the top k are selected with the largest eigen values which form the new basis of the data. Lets say X is of shape $[m \times n]$ and the the new eigen vector is of shape $[n \times k]$.
4. **Project Data:** Finally, the data is projected onto these principal components, resulting in a lower-dimensional representation which is $[m \times k]$.
   $$X_{new} = XW$$
   Remember the main idea of PCA is to find the directions in which the data varies the most, then re-express the data using those directions.

### 4.2.3 t-SNE
### 4.2.4 LDA

## 4.3 Variational Auto Encoders (VAE)

## 4.4 Generative Adversarial Networks (GAN)
GANs consist of two neural networks: a Generator and a Discriminator, trained simultaneously in a adversarial setup. The Generator creates fake data (e.g., images) from random noise, while the Discriminator tries to distinguish between real data (from the true dataset) and the fake data produced by the Generator. The magic happens because they improve together.

## 4.5 Diffusion Models