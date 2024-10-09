
# **Supervised Learning Techniques**

## **Linear Models**

### **Linear regression**

objective → find the hypothesis function → gives the target value

optimize → hypothesis function based on →  simple weighted sum of input features

loss/Error  → should be used for optimization → MSE, MAE etc

Equation for linear regression → $y = \beta_0 + \beta_1 x + \epsilon$

#### **How to get values of the coefficients (weights)**

##### **Normal equation**

$w = (X^TX)^{-1}X^Ty$

##### **Singular Value Decomposition (SVD)**

as computation complexity of $(X^TX)^{-1}$ is huge and inverse for some matrix cannot be found

##### **Gradient Descent**

Types of gradient descent

1. Batch gradient Descent → for full dataset
2. Stochastic gradient descent → for single random instance
3. mini-batch gradient descent → for small parts of dataset

Steps in gradient descent

1. start by filling θ with random values (this is called random initialization).
2. Do predictions and compute the loss from the cost Function → depending on the type of GD this can be for a single instance, a batch or the whole training set
3. Compute the Gradient (derivative of the cost function that is being used).
4. Update the Parameters.
5. Repeat Until Convergence.

The derivative of the cost function with respect to each parameter tells us how the cost function changes as we change each parameter. Specifically, it indicates the direction in which we should adjust the parameters to decrease the loss.

Here is an example of SGD. Considering only two features and MSE loss.

$y = w_1x_1 + w_2x_2 + b$

$L = \frac{1}{2}(y_{pred} - y_{true})^2$

let’s assume we have training example → $(x_1, x_2, y_{true}) = (1, 2, 3)$

Initialize the weights $w_1 = 0.1$, $w_2 = 0.2$ and $b = 0.3$

so $y_{pred} = w_1x_1 + w_2x_2 + b$
 = 0.1 * 1 + 0.2 * 2 + 0.3 = 0.8

therefore L = $\frac{1}{2}(3 - 0.8)^2 = 2.42$

Computing the gradients 

$\frac{\delta L}{\delta w_1} = \frac{\delta}{\delta w_1} \frac{1}{2}(y_{pred} - y_{true})^2$

$=  2 . \frac{1}{2}.(y_{pred} - y_{true}) ( \frac{\delta y_{pred}}{\delta w_1} - \frac{\delta y_{true}}{\delta w_1})$

$= (y_{true} - y_{pred}).\frac{\delta y_{pred}}{\delta w_1}$  

$= (y_{true} - y_{pred})\frac{\delta}{\delta w_1} (w_1x_1 + w_2x_2 + b)$

$= (y_{true} - y_{pred}). x_1$

So it is essentially multiplication of the loss with respect to the weight in the updation step, this applies for each variable

If it were batch gradient descent we would calculate $= (y_{true} - y_{pred}). x_1$ for all the x instances in the batch and divide it by the batch size.

Therefore 

$\frac{\delta L}{\delta w_1} = -2.2$     

$\frac{\delta L}{\delta w_2} = -4.4$ 

Updating the weights

if $\alpha = 0.01$

$w_1 = w_1 - \alpha \frac{\delta L}{\delta w_1} = 0.122$

$w_2 = w_2 - \alpha \frac{\delta L}{\delta w_2} = 0.244$

### **Polynomial regression**

What if data is more complex than a straight line?

add power of each feature as a new feature → using a linear model to fit non-linear data 

### **Regularization**

reduces overfitting → that can be caused by outliers

Must scale the data before regularization as it is sensitive to scale of input features.

The point is to show a bloated loss figure so that we decrease the weight by a larger value than we were previously doing

Basically the value of MSE increase by adding a certain regularization parameter to the MSE. This will also increase the value of derivation of MSE, which should reduce the weights of the parameters by a slightly larger value during the update parameters step when compared to not using regularization.

How does L1 and L2 reduce the different weights present in the model?

https://www.quora.com/How-does-the-L1-regularization-method-help-in-feature-selection

To understand how L1 helps in feature selection, you should consider it in comparison with L2.

- L1’s penalty: $\sum w_i$ → derivation of this → 1
- L2’s penalty: $\sum w_i^2$ → derivation of this → $2 * w_i$  - 1

Observation:
 L1 penalizes weights equally regardless of the magnitude of those weights. L2 penalizes bigger weights more than smaller weights.

For example, suppose $w_3 = 100$ and  $w_4=10$

- For L1 regularization both $w_3$ and $w_4$ will have the same penalty i.e. 1.
- For L2 regularization the penalty for $w_3$ will be 199 but $w_4$ will be 19.

In general, when a weight $w_i$ has already been small in magnitude, L2 does not care to reduce it to zero, L2 would rather reduce big weights than eliminate small weights to 0. The result is that the weights are reduced, but almost never reduced to 0, i.e. almost never be completely eliminated, meaning no feature selection. On  the other hand, L1 cares about reducing big weights and small weights equally. For L1, the less informative features get reduced. Some features may get completely eliminated by L1, thus we have feature selection.

Tl;DR

Increasing the MSE will also increase the weights to be decreased at each instance of gradient descent.

A model might learn the noise in the dataset, regularization discourages the model from fitting the training data too closely.

**Ridge regression (L2)**

$L = \frac{1}{m} \sum_{i=1}^m(y_{true} - y_{pred}) + \lambda \sum_{j=1}^n (w_j^2)$

Better suited for preventing overfitting and providing numerical stability

**Lasso regression (L1)**

$L = \frac{1}{m} \sum_{i=1}^m(y_{true} - y_{pred}) + \lambda \sum_{j=1}^n |w_j|$

Better suited for feature selection and preventing overfitting

**Elastic Net regression (Both)**

weighted sum of L1 and L2 regularization

$L = \frac{1}{m} \sum_{i=1}^m(y_{true} - y_{pred}) + \lambda_1 \sum_{j=1}^n (w_j^2) + \lambda_2 \sum_{j=1}^n |w_j|$

**Early stopping**

    storing the weights for the lowest RMSE on the validation data set.
    
    This is also a way of regularization.

## **Classification models**

### **Logistic regression aka binary classification**

objective → find the hypothesis function  and pass the output of the hypothesis function through a activation function (introduce non-linearity) → get target value

optimize → activation functions output of hypothesis function based on →  simple weighted sum of input features

loss/Error  → should be used for optimization → Log loss, maximum likelihood etc

****

Equation for Logistic regression (Sigmoid) → $\hat{y} = \frac{1}{e^{-(\beta_0 + \beta_1 x + \epsilon)}}$

How Sigmoid function squeezes the input values between 0 and 1?

$Sigmoid(x) = \frac{1}{1 + e^{-x}}$

value of e → euler’s constant → 2.7182818

1. x can be a positive or a negative number
2. if x is very large $e^{-x}$ becomes very small (close to 0), and 1 + 0 becomes 1 so the reciprocal of 1 is 1.
3. If x is very small $e^{-x}$ becomes very large so the sum $1 + e^{-x}$ becomes very large and the reciprocal is close to 0.
4. Therefore for a large value of x the function becomes 1 and for negative value the function becomes 0, the function output is 0.5 for value of x around 0.

x → -ve → close to 0   |    x → 0 → close to 0.5    |    x → +ve  → close to 1

****

Binary classification loss function 

$LOSS = -\frac{1}{N}\sum_{i=1}^N [y_i log(\hat{y_i}) + (1 - y_i) log(1 - \hat{y_i})]$

$\hat{y_i}$ is predicted value (0 or 1).

$y_i$ is actual value (0 or 1).

If $\hat{y_i}$  is close to 1 → $log(\hat{y_i})$ is close to 0  |  If $\hat{y_i}$  is close to 0 → $log(\hat{y_i})$ is high -ve number → check plot of $log_{10}x$ on google to understand [only between 0 and 1]

If $\hat{y_i}$  is close to 0 →  $log(1 - \hat{y_i})$ will be close to 1  |   $\hat{y_i}$  is close to 1 →  $log(1 - \hat{y_i})$ will be a high -ve number

- If $y_i$ is 1 and $\hat{y_i}$  is close to 1,  $log(\hat{y_i})$ is close to 0, resulting in a low loss.
- If $y_i$ is 1 and $\hat{y_i}$  is close to 0,  $log(\hat{y_i})$ is very negative, resulting in a high loss.
- If $y_i$ is 0 and $\hat{y_i}$  is close to 0,  $log(\hat{y_i})$ is close to 0, resulting in a low loss.
- If $y_i$ is 0 and $\hat{y_i}$  is close to 1,  $log(\hat{y_i})$ is very negative, resulting in a high loss.

Finally Averaging the loss across the whole dataset gives an estimate of how well is our model doing.

****

Evaluation of Classfication model

True Positive (TP), True Negative (TN), False Positive (FP) and False Negative (FN)

Accuracy = $\frac{TP + TN}{TP+TN+FP+FN}$ (how often are models prediction and correct)

Precision = $\frac{TP}{TP + FP}$ (how many positive predictions are actually correct)  → model’s output positivity

Recall = $\frac{TP}{TP + FN}$ (how many actual positives were predicted correctly)  → actual data positivity

F1 score = $2 \times \frac{Precision \times Recall}{Precision + Recall}$ (harmonic mean of precision and recall)

AUC-ROC → True positive rate (Recall) vs False positive rate for different threshold (default = 0.5)

AUC-PR → Precision (y-axis) vs recall (x-axis) curve (should be close to 1)

### **Softmax regression aka multi-class classification**

support multiple classes

lets say there are k classes instead of 2. In this case we use softmax function instead of log loss.

$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^Ke^z_j}$ 

$z_i$ =Wx + b

So here instead of single set of weights and bias we have same number of set of weights and bias as the number of classes

so for 3 classes 

$z_0 = W_0X +b_0$

$z_1 = W_1X + b_1$

$z_2 = W_2X + b_2$            

and so on if more classes are there

here instead of sigmoid function we use the softmax function as the activation function

**Loss Function**

$LOSS = - log(P(y|X)) = -log(\frac{e^{z_y}}{\sum_{j=1}^Ke^z_j})$

This is called categorical cross entropy loss.

How it works?

1. Compute logits
    
    z =Wx + b
    
    z = [$z_0, z_1, z_2$]  for 3 classes are computed
    
2. Applying softmax
    
    Lets say  
    
    P(y=0 ∣ x) = $\frac{e^{z_{0}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ 
    
    P(y=1 ∣ x)= $\frac{e^{z_{1}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ 
    
    P(y=2 ∣ x)= $\frac{e^{z_{2}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ 
    
3. Extract the Probability for the True Class
    
    say values of $e^{z_{2}} = 2.0, e^{z_{2}} = 1.0  \ and \ e^{z_{2}} = 0.1$
    
    P(y=0 ∣ x) = $\frac{e^{z_{0}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ =  $\frac{e^{2.0}}{e^{2.0}+ e^{1.0}+ e^{0.1}}$ = 0.59
    
    P(y=1 ∣ x)= $\frac{e^{z_{1}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ = $\frac{e^{1.0}}{e^{2.0}+ e^{1.0}+ e^{0.1}}$ = 0.242
    
    P(y=2 ∣ x)= $\frac{e^{z_{2}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ =$\frac{e^{0.1}}{e^{2.0}+ e^{1.0}+ e^{0.1}}$ = 0.009
    
    Remember    →    P(y=0 | x) + P(y=1 | x) + P(y=2 | x) = 1
    
4. Compute the Cross-Entropy Loss for the Single Example
    
    This is only done for the correct label of y, here y should be 1
    
    therefore $LOSS = -log(P(y = 1|x)) = -log(0.242) = -(-1.418) = 1.418$
    

Total Loss = $\frac{1}{N} \sum_{i=1}^N -log(P(y^i | x^i))$    So we only calculate loss for that target y value only

### **Support Vector Machine (SVM)**

Fitting the widest possible street between classes - basically the best hyperplane that is being used to separate the instances of the two classes (line → plane → hyperplane)

1. linear SVM

    easy to find the line, plane or hyperplane
    
    Large margin classification
    
    support vectors are instances at the edge of the street

2. Non-linear SVM

    Not easy to find the line, plane or hyperplane so we use the kernel trick

    polynomial kernel
    
    kernel trick - without adding higher degree features get the same results
    
    Similarity Features

Types of SVM classification

1. soft margin classification

2. hard margin classification - all instances must be off the street

### **Decision trees**

objective → At each node find a feature that best separates the dataset (most homogeneously)

optimize → least gini impurity or entropy at each node for the selected feature in case of classification or least MSE in case of linear regression

loss/Error  → should be used for optimization → Gini impurity, Entropy, MSE etc

Steps in training a decision tree

1. Initialize
    
    Start with the entire dataset at the root of the tree.
    
2. Splitting criteria
    
    For each node, evaluate all possible splits for each feature. For a numeric feature, this involves evaluating splits at every distinct value. For a categorical feature, evaluate splits based on subsets of categories.
    
3. Calculate Impurity/Variance
    
    Calculate the Gini impurity for each possible split:
    
    Gini Impurity = $1 - \sum_{i=1}^{C}p_i^2$    → for classification
    
    each node has a gini (impurity means how many instances don’t follow a particular nodes rule)
    
    where $p_i$  is the probability of a sample belonging to class i and C is the total number of classes.
    
    Entropy = $-\sum_{i=1}^{C}p_ilog_2(p_i)$    → for classification
    
     where $p_i$ is the probability of a sample belonging to class i.
    
4. Select the Best Split
    
    Choose the split that results in the lowest weighted impurity (for classification) 
    
    Weighted Impurity = $\frac{N_{left}}{N}Impurity+ \frac{N_{right}}{N}Impurity$
    
    Where N is the total number of samples in the node, $N_{left} \ and \ N_{right}$  are the number of samples in the left and right child nodes, respectively.
    
5. Split the Node
    
    Create two child nodes based on the selected split.
    
    Assign the data points to the appropriate child node based on the split criterion.
    
6. Stopping Criteria
    
    Maximum tree depth is reached.
    
7. Repeat Recursively
    
    Apply the same process recursively to each child node, treating each child node as a new parent node.
    

Branches represent decision 

leaves represent final output or classification outcome

feature scaling is not required mostly and can lead to decrease in performance

### **Naive Bayes**

Bayes' Theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event.

$P(A|B) = \frac{P(B|A).P(A)}{P(B)}$
In a classification context, we aim to find the class C that maximizes P(C∣X), where X is the set of features. 

$P(C|X) = \frac{P(X|C).P(C)}{P(X)}$

this can be simplified to 

$P(C|X) = {P(X|C).P(C)}$

The naive assumption is that the feature is conditionally independent to the class

$P(X|C) = P(x_1,x_2,...x_n|C) = P(x_1|C).P(x_2|C)...P(x_n|C)$

Example

| Email | Contains "Buy" | Contains "Cheap" | Contains "Click" | Contains "Limited" | Class |
| --- | --- | --- | --- | --- | --- |

| 1 | Yes | No | Yes | No | Spam |
| --- | --- | --- | --- | --- | --- |

| 2 | No | No | No | Yes | Not Spam |
| --- | --- | --- | --- | --- | --- |

| 3 | Yes | Yes | Yes | No | Spam |
| --- | --- | --- | --- | --- | --- |

| 4 | No | Yes | No | No | Not Spam |
| --- | --- | --- | --- | --- | --- |

| 5 | Yes | Yes | Yes | Yes | Spam |
| --- | --- | --- | --- | --- | --- |
1. Calculating prior probabilities

P(spam) = 0.6

P(not spam) = 0.4

1. Calculate likelihoods
    
    Calculate $P(x_i∣Spam)$ and $P(x_i|Not  Spam)$ for each feature.
    
    | Feature | P(Yes|Spam) | P(No|Spam) | P(Yes|Not Spam) | P(No|Not Spam) |
    |-----------------|-------------|-----------|-----------------|----------------|
    | Contains "Buy" | 3/3 = 1.0 | 0/3 = 0 | 0/2 = 0 | 2/2 = 1.0 |
    | Contains "Cheap"| 2/3 = 0.67 | 1/3 = 0.33| 1/2 = 0.5 | 1/2 = 0.5 |
    | Contains "Click"| 2/3 = 0.67 | 1/3 = 0.33| 0/2 = 0 | 2/2 = 1.0 |
    | Contains "Limited"| 1/3 = 0.33| 2/3 = 0.67| 1/2 = 0.5 | 1/2 = 0.5 |
    
2. Classify new mails
    
    We can use this mail to now check the $P(C|X) = {P(X|C).P(C)}$ probability of a class occurring.

### **Ensemble models**

Wisdom of the crowd

Aggregating the predictions of a group of predictors is called an ensemble model

Ensemble models

#### **Bagging (Bootstrap aggregating)**

same training model but different random subsets of training data for each predictor

**sampling with replacement** → after a datapoint is chosen to be a part of the sample it is replaced back into the dataset to be picked again in subsequent draws.

**Out-of-bag (OOB) evaluation**

It can be shown mathematically that 67% of the training instances are used by bagging (with replacement) and rest 33% are not used (for a single classfier but i can be used by other classifiers in the ensemble).

This 33% can be used as testing data. With enough estimators the whole training data can be then used as testing data also

one example is random forest, Xtra trees

#### **Pasting**

same training model but different random subsets of training data for each predictor
    
**sampling without replacement** → after a datapoint is chosen to be a part of the sample it is cannot be replaced back into the dataset to be picked again in subsequent draws.

example:- same like bagging example random forest but the data is without replacement

#### **Boosting**

Boosting focuses on training models sequentially, where each new model attempts to correct the errors of its predecessor. This technique aims to reduce bias and improve model accuracy.

Process

1. Train an initial base model on the entire training data.
2. Evaluate the model and increase the weight of incorrectly predicted instances.
3. Train a new model using the updated weights.
4. Repeat the process, combining the models in a weighted manner (e.g., weighted majority voting).

Types of boosting 

1. Adaboost (Adaptive boosting)
    
    Adjusts the weights of incorrectly classified instances and train the new models on updated weights.
    
2. Gradient boosting
    
    Fits new model to the residual errors of previous models.
    
    More flexible than adaboost and can handle non-linear relationships, better for higher dimensional data
    
3. XGBoost (Extreme Gradient Boosting)
    
    An optimized and efficient implementation of gradient boosting. Has parallel processing capabilities.
    
4. CatBoost (Categorical Boosting)
    
    Optimal for categorical data. No need of one-hot-encoding.
    
5. LightGBM
    
    Tree based learning algorithms and one-side sampling technique.

#### **Stacking**

Stacking involves training multiple different types of models and then using another model to combine their predictions. This technique leverages the strengths of various base models. So the underlying model can be thought of as base models and then we can use a simple Logistic regression model to combine the result of these model to get us an output.
    
Process:

1. Train several different base models on the training data.
2. Use the predictions of these base models as input features for a second-level model (meta-model).
3. Train the meta-model to make the final prediction based on the outputs of the base models.

examples are LR, Decision tree and SVM for base models and LR for meta models

#### **Voting classifiers**

Voting ensembles combine the predictions of multiple models by voting, either through majority voting for classification or averaging for regression.

1. Hard Voting
    
    Each model votes for a class, and the class with the majority votes is the final prediction.
    
2. Soft Voting
    
    Each model provides a probability for each class, and the class with the highest average probability is the final prediction.

### **Knn Algorithm**

**Choosing K:**
  
    Select the number of neighbors (K). This is a crucial hyperparameter that determines the number of nearest neighbors to consider when making a prediction.

**Calculating Distance:**
  
    For a given input, calculate the distance between this input and all the points in the training dataset. Common distance metrics include Euclidean distance, Manhattan distance, and Minkowski distance.
      
**Identifying Neighbors:**
  
    Identify the K closest neighbors to the input based on the calculated distances.
      
**Making Predictions:**
  
**For Classification:**
        
    The input is assigned to the class most common among its K nearest neighbors (majority voting).

**For Regression:**

    The input's predicted value is the average (or sometimes weighted average) of the values of its K nearest neighbors.

Main disadvantages

    considered lazy because → does not scale well mainly because it memorizes the entire dataset and performs actions based on the dataset
    
    curse of dimensionality
    
    prone to overfitting 

### **95% confidence interval for the test set accuracy of a classification model**

A 95% confidence interval (CI) for a parameter, such as test set accuracy, provides a range within which we can be 95% confident that the true value of the parameter lies.

**Step 1: Collect the Predictions and Actual Values**

Make predictions on the test set and compare them to the actual labels to obtain the number of correct and incorrect predictions.

**Step 2: Calculate the Test Set Accuracy**

$Accuracy = \frac{Number \ of \ Correct \ Predictions}{Total \ number \ of \  predictions}$

**Step 3: Calculate the Standard Error of the Accuracy**

$SE = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$

here $\hat{p}$ is the accuracy

and n is the number of predictions

**Step 4: Calculate the Confidence Interval**

$CI = \hat{p} \pm z.SE$

z → value for desired confidence interval → for 95% → z $\approx$ 1.96

### **Dimensionality Reduction**

high-dimensional datasets are at risk of being very sparse 

reduce number of features in dataset while preserving as much information as possible

Approaches of DR

1. **Projection**
    
Projection methods reduce the dimensionality of the data by transforming it onto a lower-dimensional subspace.

**PCA**

PCA is a popular linear projection method.  

It identifies the hyperplane that lies closest to the data, and then it projects the data onto it.

Steps in PCA

1. Standardize the Data
    
    Ensure the data has zero mean and unit variance. This is crucial as PCA is affected by the scale of the features.
    
    $X_{standardized} = \frac{X-\mu}{\sigma}$
    
    where μ is the mean and σ is the standard deviation of the features.
    
    Let’s take an example of 3 features
    
    Example-
    
    $X_{standardized} = \begin{bmatrix}
    1.2 & -0.9 & 2.1\\
    0.8 & -1.1 & 2.5\\
    1.0 & -1.0 & 2.3
    \end{bmatrix}$
    
    *Standardizing the data ensures that each feature contributes equally to the PCA computation.*
    
2. Compute the Covariance Matrix
    
    The covariance matrix captures the pairwise correlations between features.
    
    $\sum = \frac{1}{n-1}X_{standardized}^TX_{standardized}$
    
    where n is the number of samples.
    
    Example-
    
    $\sum = \begin{bmatrix}
    0.02 & 0.001 & 0.04\\
    0.001 & 0.03 & 0.002\\
    0.03 & 0.002 & 0.006
    \end{bmatrix}$
    
    *The covariance matrix represents the covariance (joint variability) between pairs of features.*
    
3. Compute Eigenvalues and Eigenvectors
    
    The eigenvalues represent the amount of variance explained by each principal component, and the eigenvectors represent the directions of these components.
    
    $\sum v = \lambda v$
    
    where λ is an eigenvalue and v is the corresponding eigenvector.
    
    Example-
    
    Suppose the eigen values are $\lambda_1$=0.07, $\lambda_2$= 0.04 $\lambda_3$= 0.003.
    
    The corresponding eigen vectors are $v_1, v_2$ and $v_3$.
    
    *The eigenvectors of the covariance matrix are the principal components. Eigenvalues correspond to the variance explained by each component.* 
    
    *For instance, the first principal component is the direction that maximizes the variance in the data.*
    
4. Sort Eigenvalues and Select Principal Components
    
    Sort the eigenvalues in descending order and select the top k eigenvectors corresponding to the largest eigenvalues. These eigenvectors form the principal components.
    
    Example-
    
    Choose the top two eigenvectors based on the eigenvalues: $v_1$ and $v_2$
    
5. Transform the Data
    
    Project the original data onto the new k-dimensional subspace.
    
    $X_{PCA} = X_{standardized} W$
    
    where W is the matrix of the selected eigenvectors.
    

PCA helps in identifying the directions of maximum variance and projecting the data onto these directions to reduce dimensionality.

2. **Manifold learning**

Manifold learning methods assume that the high-dimensional data lies on a low-dimensional manifold within the higher-dimensional space. 

Popular manifold structure

1. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
2. **Isomap**
3. **Locally Linear Embedding (LLE)**











