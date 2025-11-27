# 1 Data analysis and statistics

## 1.1 Bias and variance

When we build a machine learning model, we want it to learn patterns from data — not too little, not too much.

**Bias**
1. Think of bias as being too simple.
2. A model with high bias makes strong assumptions and ignores details.
3. Example: If you try to fit a straight line to data that’s actually curved, your model will miss patterns → underfitting.

**Variance**
1. Variance is the opposite: being too sensitive.
2. A model with high variance tries to follow every tiny wiggle in the data, including noise.
3. Example: If you draw a super squiggly line that perfectly touches every point, it won’t work well on new data → overfitting.

**The Tradeoff**
1. If bias is too high → the model is too simple → poor performance.
2. If variance is too high → the model is too complex → poor generalization.
3. The sweet spot is in the middle: a model that’s just complex enough to capture the true patterns but not the noise.

**In short**
1. Bias = error from being too simple.
2. Variance = error from being too complex.
3. The tradeoff = finding balance so the model learns the real signal, not the noise.

## 1.2 Statistics

1. Mean
2. Mode
3. Median
4. Variance
5. Covariance
6. Standard deviation
7. Skewness
8. Kurtosis
9. Correlation
10. IQR
11. Range

## 1.3 Similarity and distance

1. Similarity
2. Cosine similarity
3. Jaccard similarity
4. Euclidean distance
5. Pearson correlation coefficient
6. Minkowski distance
7. Manhattan distance
8. Hamming distance