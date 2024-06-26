# Machine learning

**Notations** 

Important - purple

questions - red

Links - green

This is the link to Excalidraw

[https://excalidraw.com/#json=i1UW2iD_kiEFwWCRqL5Uq,b1nCADVf58OCrjqzk9GHGg](https://excalidraw.com/#json=i1UW2iD_kiEFwWCRqL5Uq,b1nCADVf58OCrjqzk9GHGg)

What is a derivative?

A derivative of a function represents the rate at which a function's value changes as its input changes. In simpler terms, the derivative measures how a function's output changes in response to changes in its input.

In simpler words let’s look at the formula

$f'(x) = \frac{df}{dx} = lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}$

Also remember the formula of slope $= \frac{y_2 - y_1}{x_2 - x_1}$

so basically the formula give you the slope of the secant line through f(x) and f(x + h)

![Untitled](Machine%20learning%202025837b28ef4aecbc1b3261dbcfb669/Untitled.png)

What are the rules of differentiation?

1. Sum rule
    
    $\frac{d}{dx}(f(x) + g(x)) = \frac{df}{dx} + \frac{dg}{dx}$
    
2. Product rule
    
    $\frac{d}{dx}(f(x).g(x)) = \frac{df}{dx}.g(x) + \frac{dg}{dx}.f(x)$
    
3. Chain rule
    
    $\frac{d}{dx}(g \circ f)(x) = \frac{d}{dx}(g(f(x)) = \frac{dg}{dx} \frac{df}{dx}$
    

Linear Algebra

vector

span

The span of two vectors v and w is the set of all their linear combinations i.e. av + bw.

matrix and linear transformation

determinant

column space 

null space

non-square matrix

Inverse matrix

Column space

Null space

eigenvector and eigenvalue

**Loss Types**

**MSE / L2**

**MAE / L1**

**RMSE**

**Huber loss = MAE + MSE**

Robust to outliers

$Huber = \frac{1}{n}\sum_{i=1}^n(y_i - \hat{y_i})^2$                                     $y_i - \hat{y_i} \le \delta$

$Huber = \frac{1}{n}\sum_{i=1}^{n}\delta (|y_i -\hat{y_i}|-\frac{1}{2}\delta)$                 $y_i - \hat{y_i} > \delta$

$\delta$ is the threshold hyperparameter parameter → you can set them manually

**Binary Cross-Entropy Loss / Log Loss** 

only for dual classes

$L(y,\hat{y}) = -\frac{1}{N}\sum_{i=1}^{N}[y_i(log(\hat{y_i})+(1-y_i)log(1-\hat{y_i})]$

**Categorical Cross-Entropy Loss**

Categorical cross entropy is a loss function used in classification problems where each input belongs to one of multiple classes → softmax

$L = -\frac{1}{m} \sum_{i=1}^{m}y_ilog(\hat{y_i})$

- m is the number of classes
- $y_i$ is the true label (one-hot encoded).
- $\hat{y_i}$ the predicted probability for example i.

example:

Suppose y = [0, 1, 0] is the actual value for three classes (A, B, C) and it belongs to class B which is represented in one hot encoded form

and model predicts $\hat{y}$ = [0.2, 0.7, 0.1]

L = - ( 0.Log(0.2) + 1.log(0.7) + 0.Log(0.1)            → yani ki jiska jitna jyada probability value x utna less log(x) value

L = -Log(0.7)

L = 0.357

**Sparse categorical cross entropy** 

Sparse categorical cross entropy is similar to categorical cross entropy but is used when the true labels are provided as **integers** instead of one-hot encoded vectors

$L = -log(\hat{y_c})$

$\hat{y_c}$ is the predicted probability for true class c

example:

y = 1 the true label is given as an integer

and model predicts $\hat{y}$ = [0.2, 0.7, 0.1]

L = -Log($\hat{y_1}$)        here is the real value of y = 0 then we would have took log of $y_0$ 

L = -log(0.7)

L = 0.357

Cross Entropy Loss takes the target labels as One Hot Encoded Vectors, and this may not be feasible if you have many target classes, it is more efficient to represent the target labels as integers rather than one-hot encoded vectors and hence this is where Sparse Cross Entropy should be used. Another caveat — Sparse Cross Entropy should be used when your classes are mutually exclusive, that is, when one sample belongs to only one class, while Cross Entropy should be used when one sample can have multiple classes or labels are soft probabilities like [0.3, 0.7, 0.1].

---

## Supervised Learning Techniques

---

### ***Linear regression***

objective → find the hypothesis function → get target value

optimize → hypothesis function based on →  simple weighted sum of input features

loss/Error  → should be used for optimization → MSE, MAE etc

How to get values of the coefficients (weights)

**Normal equation**

$w = (X^TX)^{-1}X^Ty$

**Singular Value Decomposition (SVD)**

as computation complexity of $(X^TX)^{-1}$ is huge and inverse for some matrix cannot be found

**Gradient Descent**

Types of gradient descent

1. Batch gradient Descent → for full dataset
2. Stochastic gradient descent → for single random instance
3. mini-batch gradient descent → for small parts of dataset

Steps in gradient descent

1. start by filling θ with random values (this is called random initialization).
2. Compute the Cost Function (MSE) → depending on the type this can be for a single instance, a batch or the whole training set
3. Compute the Gradient (derivative of the cost function that is being used).
4. Update the Parameters.
5. Repeat Until Convergence.

The derivative of the cost function with respect to each parameter tells us how the cost function changes as we change each parameter. Specifically, it indicates the direction in which we should adjust the parameters to decrease the cost function.

Here is an example of SGD

$y = w_1x_1 + w_2x_2 + b$

$L = \frac{1}{2}(y_{true} - y_{pred})^2$

let’s assume we have training example → $(x_1, x_2, y_{true}) = (1, 2, 3)$

Initialize the weights $w_1 = 0.1$, $w_2 = 0.2$ and $b = 0.3$

so $y_{pred} = w_1x_1 + w_2x_2 + b$
 = 0.1 * 1 + 0.2 * 2 + 0.3 = 0.8

therefore L = $\frac{1}{2}(3 - 0.8)^2 = 2.42$

Computing the gradients 

$\frac{\delta L}{\delta w_1} = \frac{\delta}{\delta w_1} \frac{1}{2}(y_{true} - y_{pred})^2$

$=  2 . \frac{1}{2}.(y_{true} - y_{pred}) ( \frac{\delta y_{pred}}{\delta w_1} + \frac{\delta y_{true}}{\delta w_1})$

 $= (y_{true} - y_{pred}).\frac{\delta y_{pred}}{\delta w_1}$ 

$= (y_{true} - y_{pred})\frac{\delta}{\delta w_1} (w_1x_1 + w_2x_2 + b)$

 $= (y_{true} - y_{pred}). x_1$

So it is essentially multiplication of the loss with respect to the weight in the updation step, this applies for each variable

Therefore 

$\frac{\delta L}{\delta w_1} = -2.2$     

$\frac{\delta L}{\delta w_2} = -4.4$ 

Updating the weights

if $\alpha = 0.01$

$w_1 = w_1 - \alpha \frac{\delta L}{\delta w_1} = 0.122$

$w_2 = w_2 - \alpha \frac{\delta L}{\delta w_2} = 0.244$

**Regularization of linear models**

reduces overfitting

Must scale the data before regularization as it is sensitive to scale of input features.

Basically the value of MSE increase by adding a certain regularization parameter to the MSE. This will also increase the value of derivation of MSE, which should reduce the weights of the parameters by a slightly larger value during the update parameters step when compared to when we were not using regularization. Thereby keeping the weights as small as possible.

How does L1 and L2 reduce the different weights present in the model?

[https://www.quora.com/How-does-the-L1-regularization-method-help-in-feature-selection](https://www.quora.com/How-does-the-L1-regularization-method-help-in-feature-selection)

To understand how L1 helps in feature selection, you should consider it in comparison with L2.

- L1’s penalty: $\sum w_i$ → derivation of this → 1
- L2’s penalty: $\sum w_i^2$ → derivation of this → $2 * w_i$  - 1

Observation:
 L1 penalizes weights equally regardless of the magnitude of those weights. L2 penalizes bigger weights more than smaller weights.

For example, suppose $w_3 = 100$ and  $w_4=10$

- By reducing $w_3$ by 1, L1’s penalty is reduced by 1. By reducing w4 by 1, L1’s penalty is also reduced by 1.
- By reducing $w_3$ by 1, L2’s penalty is reduced by 199. By reducing $w_4$ by 1, L2’s penalty is reduced by only 19. Thus, L2 tends to prefer reducing w3 over w4.

In general, when a weight $w_i$ has already been small in magnitude, L2 does not care to reduce it to zero, L2 would rather reduce big weights than eliminate small weights to 0. The result is that the weights are reduced, but almost never reduced to 0, i.e. almost never be completely eliminated, meaning no feature selection. On  the other hand, L1 cares about reducing big weights and small weights equally. For L1, the less informative features get reduced. Some features may get completely eliminated by L1, thus we have feature selection.

Tl;DR

Increasing the MSE will also increase the weights to be decreased at each instance of gradient descent.

Ridge regression (L2)

$L = \frac{1}{m} \sum_{i=1}^m(y_{true} - y_{pred}) + \lambda \sum_{j=1}^n (w_j^2)$

Lasso regression (L1)

$L = \frac{1}{m} \sum_{i=1}^m(y_{true} - y_{pred}) + \lambda \sum_{j=1}^n |w_j|$

Elastic Net regression (Both) 

weighted sum of L1 and L2 regularization

$L = \frac{1}{m} \sum_{i=1}^m(y_{true} - y_{pred}) + \lambda_1 \sum_{j=1}^n (w_j^2) + \lambda_2 \sum_{j=1}^n |w_j|$

**Early stopping**

storing the weights for the lowest RMSE on the validation data set.

This is also a way of regularization.

---

### **Polynomial regression**

What if data is more complex than a straight line?

add power of each feature as a new feature → using a linear model to fit non-linear data 

---

### Classification

Linear classification

Non-linear classification

1. If some dataset is not linearly separable you can always add polynomial features resulting in linearly separable dataset.
2. Add similar computed features.

### **Logistic regression**

objective → find the hypothesis function  and pass the output of the hypothesis function through a activation function (sigmoid) → get target value

optimize → activation functions output of hypothesis function based on →  simple weighted sum of input features

loss/Error  → should be used for optimization → Log loss, maximum likelihood etc

How Sigmoid function squeezes the input values between 0 and 1?

$Sigmoid(x) = \frac{1}{1 + e^{-x}}$

value of e → euler’s constant → 2.7182818

1. x can be a positive or a negative number
2. if x is very large $e^{-x}$ becomes very small (close to 0), and 1 + 0 becomes 1 so the reciprocal of 1 is 1.
3. If x is very small $e^{-x}$ becomes very large so the sum $1 + e^{-x}$ becomes very large and the reciprocal is close to 0.
4. Therefore for a large value of x the function becomes 1 and for negative value the function becomes 0, the function output is 0.5 for value of x around 0.

x → -ve → close to 0   |    x → 0 → close to 0.5    |    x → +ve  → close to 1

**Loss function**

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

**Evaluation of Logistic regression**

True Positive (TP), True Negative (TN), False Positive (FP) and False Negative (FN)

Accuracy = $\frac{TP + TN}{TP+TN+FP+FN}$ (how often are models prediction and correct)

Precision = $\frac{TP}{TP + FP}$ (how many positive predictions are actually correct)  → model’s ouput positivity

Recall = $\frac{TP}{TP + FN}$ (how many actual positives were predicted correctly)  → actual data positivity

F1 score = $2 \times \frac{Precision \times Recall}{Precision + Recall}$ (harmonic mean of precision and recall)

AUC-ROC → True positive rate (Recall) vs False positive rate (should be close to 1)

AUC-PR → Precision (y-axis) vs recall (x-axis) curve (should be close to 1)

---

### 95% confidence interval for the test set accuracy of a classification model

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

---

### **Softmax regression**

support multiple classes

lets say there are k classes instead of 2 (binomial).

$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^Ke^z_j}$ 

$z_i$ =Wx + b

So here instead of single set of weights and bias we have same number of set of weights and bias as the number of classes

so for 3 classes →       $z_0 = W_0X +b_0$           |         $z_1 = W_1X + b_1$            |     $z_2 = W_2X + b_2$            and so on if more classes are there

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
    
    P( y=0 ∣ x) = $\frac{e^{z_{0}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ 
    
    P(y=1 ∣ x)= $\frac{e^{z_{1}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ 
    
    P(y=2 ∣ x)= $\frac{e^{z_{2}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ 
    
3. Extract the Probability for the True Class
    
    say values of $e^{z_{2}} = 2.0, e^{z_{2}} = 1.0  \ and \ e^{z_{2}} = 0.1$
    
    P( y=0 ∣ x) = $\frac{e^{z_{0}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ =  $\frac{e^{2.0}}{e^{2.0}+ e^{1.0}+ e^{0.1}}$ = 0.59
    
    P(y=1 ∣ x)= $\frac{e^{z_{1}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ = $\frac{e^{1.0}}{e^{2.0}+ e^{1.0}+ e^{0.1}}$ = 0.242
    
    P(y=2 ∣ x)= $\frac{e^{z_{2}}}{e^{z_{0}} + e^{z_{1}} + e^{z_{2}}}$ =$\frac{e^{0.1}}{e^{2.0}+ e^{1.0}+ e^{0.1}}$ = 0.009
    
    Remember    →    P(y=0 | x) + P(y=1 | x) + P(y=2 | x) = 1
    
4. Compute the Cross-Entropy Loss for the Single Example
    
    This is only done for the correct label of y, here y should be 1
    
    therefore $LOSS = -log(P(y = 1|x)) = -log(0.242) = -(-1.418) = 1.418$
    

Total Loss = $\frac{1}{N} \sum_{i=1}^N -log(P(y^i | x^i))$    So we only calculate loss for that target y value only

---

### Support Vector Machine (SVM)

widest possible street between classes - basically the best hyperplane that is being used to separate the instances of the two classes (line → plane → hyperplane)

soft margin classification

hard margin classification - all instances must be off the street

linear SVM

easy to find the line, plane or hyperplane

Large margin classification

support vectors are instances at the edge of the street

Non-linear SVM

Not easy to find the line, plane or hyperplane so we use the kernel trick

polynomial kernel

kernel trick - without adding higher degree features get the same results

Similarity Features

---

### Decision trees

objective → At each node find a feature that best separates the dataset (most homogeneously)

optimize → least gini impurity or entropy at each node for the selected feature in case of classification or least MSE in case of linear regression

loss/Error  → should be used for optimization → Gini impurity, Entropy, MSE etc

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

feature scaling is not required mostly

---

### Naive Bayes

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
    

---

### Ensemble models and random forest

Wisdom of the crowd

Aggregating the predictions of a group of predictors is called an ensemble model

**Random forest**

ensemble of decision trees only.

Ensemble types are as follows

1. **Bagging (Bootstrap aggregating)**
    
    same training model but different random subsets of training data for each predictor
    
    sampling with replacement → after a datapoint is chosen to be a part of the sample it is replaced back into the dataset to be picked again in subsequent draws.
    
    **Out-of-bag (OOB) evaluation**
    
    It can be shown mathematically that 67% of the training instances are used by bagging (with replacement) and rest 33% are not used (for a single classfier but i can be used by other classifiers in the ensemble).
    
    This 33% can be used as testing data. With enough estimators the whole training data can be then used as testing data also
    
2. **Pasting**
    
    same training model but different random subsets of training data for each predictor
    
    sampling without replacement → after a datapoint is chosen to be a part of the sample it is cannot be replaced back into the dataset to be picked again in subsequent draws.
    
3. **Boosting**
    
    Boosting focuses on training models sequentially, where each new model attempts to correct the errors of its predecessor. This technique aims to reduce bias and improve model accuracy.
    
    Process
    
    1. Train an initial base model on the entire training data.
    2. Evaluate the model and increase the weight of incorrectly predicted instances.
    3. Train a new model using the updated weights.
    4. Repeat the process, combining the models in a weighted manner (e.g., weighted majority voting).
    
    Types of boosting 
    
    1. Adaboost (Adaptive boosting)
        
        Adjusts the weights of incorrectly classified instances and combines the models with weights based on their accuracy.
        
    2. Gradient boosting
        
        Sequentially adds models to minimize a specified loss function, usually by fitting models to the residual errors of the previous models.
        
    3. XGBoost (Extreme Gradient Boosting)
        
        An optimized and efficient implementation of gradient boosting.
        
4. **Stacking**
    
    Stacking involves training multiple different types of models and then using another model to combine their predictions. This technique leverages the strengths of various base models.
    
    Process:
    
    1. Train several different base models on the training data.
    2. Use the predictions of these base models as input features for a second-level model (meta-model).
    3. Train the meta-model to make the final prediction based on the outputs of the base models.
5. **Voting classifier**
    
    Voting ensembles combine the predictions of multiple models by voting, either through majority voting for classification or averaging for regression.
    
    1. Hard Voting
        
        Each model votes for a class, and the class with the majority votes is the final prediction.
        
    2. Soft Voting
        
        Each model provides a probability for each class, and the class with the highest average probability is the final prediction.
        

---

### Dimensionality Reduction

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

---

## Unsupervised Learning Techniques

---

 The vast majority of the available data is unlabeled

We have the input features X, but we do not have the labels y

Unsupervised learning tasks → clustering, anomaly detection and Density estimation

---

### k-means

given all the instance labels, locate each cluster’s centroid by computing the mean of the instances in that cluster - both centroids and labels are not given

Process

1. place the centroids randomly (randomly select k instances from the training set).
2. Assign each data point to the nearest centroid creating k clusters (euclidean distance).
3. Recalculate the mean of all the data points for each cluster and assign the new point as the centroid. 
4. Repeat 2,3 and 4 till the centroids stop moving.

Clustering types

Hard clustering 

directly assign the cluster for a node

Soft clustering

score per cluster

Centroid initialization methods

Setting the init hyperparameter (if you have an idea of where the centroid should be)

Run algorithm multiple times (with different random initialization for the centroid)

Performance metric 

model’s inertia → sum of the squared distances between the instances and their closest centroids.

---

### k-means++

smarter initialization step that tends to select centroids that are distant from one another

1. Randomly select the first centroid, say $\mu_1$.
2.  Calculate the distance of all points from  $\mu_1$, then select the second centroid  $\mu_2$ with a probability proportional to sum of the squared distances.
    
    Let’s say the distance to the few point from the current centroid is as follows
    
    - $D(x_1)$ = 1
    - $D(x_2)$ = 2
    - $D(x_3)$ = 3
    - $D(x_4)$ = 4
    
    Squaring these distances, we get:
    
    - $D(x_1)^2$ = 1
    - $D(x_2)^2$ = 4
    - $D(x_3)^2$ = 9
    - $D(x_4)^2$ =16
    
    The sum of the squared distances is 1+4+9+16=301+4+9+16=30. The probabilities for each point being selected as the next centroid are:
    
    - $P(x_1)$ = 1/30 ≈ 0.033
    - $P(x_2)$ = 4/30 ≈ 0.133
    - $P(x_3)$ = 9/30 = 0.3
    - $P(x_4)$ = 16/30 ≈ 0.533
    
    So basically we do not always use the farthest point but as the distance increases the probability increases too .
    

---

### Accelerated and mini-batch k-means

1. **Accelerated k-Means**

Accelerated k-Means algorithms aim to speed up the standard k-means clustering process, which can be computationally intensive due to the need to repeatedly compute distances between data points and cluster centroids.

1. Elkan's Algorithm
    
    Elkan's algorithm speeds up k-means by reducing the number of distance calculations needed during each iteration. 
    
    1. **Initialization**: Start with initial centroids, like in standard k-means.
    2. **Distance Bounds**: Maintain upper and lower bounds for the distances between each point and the centroids.
    3. **Update Centroids**: After assigning points to the nearest centroids, update the centroids.
    4. **Bounds Update**: Update the bounds based on the new centroids.
    5. **Pruning**: Use the bounds to skip distance calculations for points that are unlikely to change their cluster assignments.
2. k-d Tree and Ball Tree Methods
    1. k-d Tree → A binary tree where each node represents a splitting hyperplane dividing the space into two subspaces.
    2. ball tree → A hierarchical data structure where data points are encapsulated in hyperspheres (balls).

1. **Mini-batch k-means**
    
    Mini-batch k-means is a variant of the k-means algorithm that reduces the computational cost by using small, random samples (mini-batches) of the dataset in each iteration rather than the entire dataset.
    
    Process
    
    1. **Initialization**: Start with initial centroids, just like in standard k-means.
    2. **Mini-Batch Selection**: Randomly select a mini-batch of data points from the dataset.
    3. **Cluster Assignment**: Assign each point in the mini-batch to the nearest centroid.
    4. **Centroid Update**: Update the centroids based on the mini-batch. The update rule is typically a weighted average to account for the small size of the mini-batch.
    5. **Repeat**: Iterate the mini-batch selection, cluster assignment, and centroid update steps until convergence or a maximum number of iterations is reached.
    

Finding the optimal number of clusters

1. Inertia is not a good performance metric when trying to choose k because it keeps getting lower as we increase k.
    
    $Inertia = \sum_{i=1}^n \sum_{j=1}^k 1(c_i=j) || x_i - \mu_j||^2$
    
    - n is the number of data points.
    - k is the number of clusters.
    - $x_i$ is the i-th data point.
    - $\mu_j$ is the centroid of the j-th cluster.
    - 1($c_i$ = j) is an indicator function that is 1 if the data point i belongs to cluster j, and 0 otherwise.
    - $||x_i -\mu_j ||^2$ is the squared Euclidean distance between the data point and the cluster centroid.
2. Silhouette score
    
    The Silhouette score is a metric used to evaluate the quality of a clustering algorithm. It measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
    
    For each data point i, the Silhouette score s(i) is calculated using the following steps
    
    1. Calculate the mean intra-cluster distance (a(i))
        
        This is the average distance between the data point i and all other points in the same cluster.
        
        $a(i) = \frac{1}{|C_i|-1}\sum_{j \in C, j\neq i} d(i,j)$
        
        where $c_i$ is the cluster containing i, and d(i,j) is the distance between points i and j.
        
    2. Calculate the mean nearest-cluster distance (b(i))
        
        This is the average distance between the data point i and all points in the nearest cluster that is not $C_i$
        
        $b(i) = min_{C_k \neq C_i} \frac{1}{|C_k|}\sum_{j \in C_k} d(i,j)$
        
        where $C_k$  is any cluster that is not $C_i$
        
    3. Calculate the Silhouette score (s(i))
        
        $s(i) = \frac{b(i) - a(i)}{max(a(i),b(i))}$
        
        The Silhouette score ranges from -1 to 1:
        
        1. s(i) close to 1 indicates that the data point is well-clustered and appropriately assigned.
        2. s(i) close to 0 indicates that the data point is on or very close to the decision boundary between two neighboring clusters.
        3. s(i) close to -1 indicates that the data point might have been assigned to the wrong cluster.

---

### DBSCAN

density-based spatial clustering of applications with noise (DBSCAN)

ε (epsilon), min_samples,  ε-neighborhood, core instance

1. For each instance, the algorithm counts how many instances are located within a small distance ε (epsilon) from it. This region is called the instance’s ε-neighborhood.
2. If an instance has at least min_samples instances in its ε-neighborhood (including itself), then it is considered a core instance.
3. All instances in the neighborhood of a core instance belong to the same cluster.
4. Any instance that is not a core instance and does not have one in its neighborhood is considered an anomaly.

---

### Gaussian Mixtures

A probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown.

---

## Artificial Neural Network

---

### The perceptron

Threshold logic unit (TLU)

1. The inputs and output are numbers (instead of binary on/off values), and each input connection is associated with a weight.
2. The TLU first computes a linear function of its inputs.
3. Then it applies a step function to the result.
4. It’s almost like logistic regression, except it uses a step function instead of the logistic (sigmoid) function.

A perceptron is composed of one or more TLUs organized in a single layer.

Remember each neuron has a bias value

---

### Backpropagation for perceptron

1. The weights of the perceptron are initialized randomly or with small random values.
2. The input data is fed into the network, and calculations are carried out layer by layer from the input layer to the output layer to produce the output.
    1. The input values are multiplied by their respective weights.
    2. These products are summed, and the sum is passed through an activation function to produce the output.
3. The error (or loss) is calculated by comparing the network’s output with  the actual target value using a loss function. A common loss function for a single output is the mean squared error (MSE)
4. Backpropagation involves three main steps
    1. Calculate the gradient: The gradient of the error with respect to each weight is calculated using the chain rule of calculus. This involves determining how changes in weights affect the error.
    2. Update the Weights:  The weights are updated in the direction that reduces the error, which is opposite to the gradient. 
    3. Iterate.

---

### Backpropagation for multi-layer perceptron

**Solved Example Back Propagation Algorithm Multi-Layer Perceptron Network by Dr. Mahesh Huddar**

[https://www.youtube.com/watch?v=tUoUdOdTkRw](https://www.youtube.com/watch?v=tUoUdOdTkRw)

Forward pass

Backward pass

Chain rule

---

### Activation functions

1. **Heaviside**: step function → 0 or 1
2. **Tanh**: S shaped → -1 to 1 → 
3. **Sigmoid**: S shaped → 0 to 1 → logistic function → if you need gradient (because step function has no gradient so no progress in gradient descent)
    
    If you want to guarantee that the output will always fall within a given range of values, then use the Sigmoid function.
    
4. **Rectified Linear Unit (ReLU)** : to  get positive output only  →  _/ shaped function
    
    If you want to guarantee that the output will always be positive, then use the ReLU activation function.
    
5. **Leaky ReLU:**  Slope z < 0 to ensure that leaky ReLU never dies.
6. **Randomized Leaky ReLU (RReLU):** The slope towards the negative side can be either 0, +ve or -ve depending on a hyperparameter.
7. **Parametric Leaky ReLU (PReLU):** The slope towards the -ve side is decided by a hyperparameter.
8.  **Exponential Linear unit (ELU):** performed better than all ReLU’s with lesser training time and better results only disadvantage being it is slower to compute than ReLU.
9. **Scaled Exponential Linear Unit (SELU):** about 1.05 times ELU
10. **Gaussian error linear unit** **(GELU):** other activation function discussed above but is computationally intensive.
11. **Swish and Mish: other** variants of ReLU. 

---

TensorFlow playground → understanding MLPs → effect of hyperparameters (number of layers, neurons, activation function and more)

---

**How to decide the number of neurons for each layer in a neural network?**

[https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3](https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3)

- The number of hidden neurons should be between the size of the input layer and the size of the output layer.
- The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
- The number of hidden neurons should be less than twice the size of the input layer.

GridSearchCV

RandomizedSearchCV

---

### The Vanishing/Exploding Gradients Problems

If weights are initialized with high variance even if i/p has low variance the o/p of a layer can have greater variance. The variance of the outputs of each layer is much greater than the variance of its inputs.

Therefore the gradient can be close to 0 leading to vanishing gradients.

So how to solve this problem?

So we have to solve 2 problems 

1. How to initialize the weights?
2. Which activation function to use?

Solving problem 1

**Glorot and He Initialization** 

For weight initialization :- 

Main idea 

1. The variance of the outputs of each layer to be equal to the variance of its inputs
2. The gradients should have equal variance before and after flowing through a layer in the reverse direction
- Fan-in: The number of input units to a layer.
- Fan-out: The number of output units from a layer.

Therefore connection weights should be initialized randomly → Glorot initialization

- **Mean**: The mean of the weights is typically 0.
- **Variance**: The variance of the weights is set to:
    
    Var(W) = $\sigma^{2} = \frac{2}{ fan_{in} + fan_{out}}$
    

This will balance the the variance between the i/p and o/p layer  uniform.

solving problem 2

Use Relu or other variants of Relu

**Batch Normalization**

However solving these two problems still does not ensure that the vanishing/exploding gradients problem does not reoccur during training. 

To address this we have batch normalization.

This is done just before or after the activation function in each layer

Steps in BN:

1. Compute the Mean and Variance: For a given mini-batch, compute the mean and variance of the activations.
    
    $\mu_{batch} = \frac{1}{m} \sum_{i=1}^{m}x_i$
    
    $\sigma_{batch}^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_{batch})^2$
    
2. Normalize the Activations: Subtract the mean and divide by the standard deviation to normalize the activations.
    
    $\hat{x_i} = \frac{x_i - \mu_{batch}}{\sqrt{\sigma_{batch}^2 + \epsilon}}$
    
3. **Scale and Shift**: Introduce two trainable parameters, γ (scale) and β (shift), to allow the model to learn the optimal scaling and shifting of the normalized activations.
    
    $y_i = \gamma \times \hat{x_i} + \beta$     → for each training instances and for each feature
    

BN performs scaling. 

BN also acts like a regularization thus eliminating the need of regularization techniques.

**Gradient Clipping**

used in RNN for exploding gradients mostly

clip the gradients during backpropagation so that they never exceed some threshold.

---

**Reusing  pretrained models**

1. Transfer learning
    
    If the task at hand is similar to a task that is already solve by a deep neural network (DNN) we can use some or many layer of the existing DNN to help increase the accuracy of the model. We can freeze the reused layers in the first few epochs so that out model are also able to learn and adjust. However it is usally very difficult to find good good configurations and is generally used only in CNNs.
    
2. Unsupervised Pretraining
    
    If you did not find any model trained on a similar task use GNN or autoencoders (RBMs)
    
3. Pretraining on an Auxiliary Task
    
    If not much labelled training data is present first train a neural network on an auxiliary task for which you can easily obtain or generate labeled training data, then reuse the lower layers of that network for your actual task.
    

---

### Faster Optimizers

Till now we have only used SGD where we simply udate the parameters based on the derivation values.
But we can speed up this process by using different optimizers

1. Momemtum
    
    A bowling ball rolling down a gentle slope on a smooth surface: it will start out slowly, but it will quickly pick up momentum until it eventually
    reaches terminal velocity.
    
    $\beta$ is a hyperparameter, $\alpha$ is the learning rate both are set during training
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) ← like this
    
    1. Initialize Velocity (accumulated gradient)
    $v_t = 0$
    2. Compute gradient
    $g_t = \nabla_\theta J(\theta_t)$
    3. Update the velocity
    $v_t = \beta v_{t-1} + (1 - \beta) g_t$
    4. Update the parameters
    $\theta_{t+1} = \theta_t - \alpha v_t$

1. Nesterov Accelerated Gradient
    
    It measures the gradient of the cost function slightly ahead in the direction of the momentum
    
    1. Initialize Velocity (accumulated gradient)
    $v_t = 0$
    2. Lookahead position
    $\theta_{lookahead} = \theta_t - \beta v_{t-1}$
    3. Compute gradient
    $g_t = \nabla_\theta J(\theta_{lookahead})$
    4. Update the velocity
    $v_t = \beta v_{t-1} + \alpha g_t$
    5. Update the parameters
    $\theta_{t+1} = \theta_t - v_t$

1. Adagrad
    
    It maintains a running sum of the squares of the gradients for each parameter. 
    
    1. Initialize Accumulated Squared Gradients
    $G_t = 0$
    2. Compute gradient
    $g_t = \nabla_\theta J(\theta_t)$
    3. Update Accumulated Squared Gradients
    $G_t = G_{t-1} + g_t^2$
    4. Update parameters
    $\theta_{t-1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}
    } \bigodot  g_t$
    
    Here, α is the initial learning rate, ϵ is a small constant to prevent division by zero, and ⊙ denotes element-wise multiplication.
    

1. RMSProp
    
    AdaGrad runs the risk of slowing down a bit too fast and never converging to the global optimum
    
    1. Initialize Accumulated Squared Gradients
    $E[g^2]_t = 0$
    $E[g^2]_t$   is the exponentially decaying average of past squared gradients at time step t.
    2. Compute gradient
    $g_t = \nabla_\theta J(\theta_t)$
    3. Update accumulated squared gradient
    $E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2$
    4. Update parameters
    $\theta_{t-1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t  + \epsilon}
    } \bigodot  g_t$
    α is the learning rate, ϵ is a small constant to prevent division by zero, and ⊙ denotes element-wise multiplication.

1. Adam
    
    a. Initialize Moment Estimates and Time Step
    
    $m_t = 0$ ( First moment estimate )
    
    $v_t = 0$  (Second moment estimate)
    
    t = 0  (Time step)
    
    b. Compute Gradient
    $g_t = \nabla_\theta J(\theta_t)$
    
    c. Update Time Step 
    
    t = t + 1
    
    d. Update Biased First Moment Estimate
    
    $m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t^2$
    
    $\beta_1$ is the decay rate for the first moment estimate (typically around 0.9).
    
    e. Update Biased Second Moment Estimate
    
    $v_t = \beta_2 c_{t-1} + (1 - \beta_2)g_t^2$
    
    $\beta_2$ is the decay rate for the second moment estimate (typically around 0.999).
    
    f. Compute Bias-Corrected First Moment Estimate
    
    $\hat{m_t} = \frac{m_t}{1 - \beta_1^t}$
    
    g. Compute Bias-Corrected Second Moment Estimate
    
    $\hat{v_t} = \frac{v_t}{1 - \beta_2^t}$
    
    h. Update Parameters
    
    $\theta_{t-1} = \theta_t - \frac{\alpha \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}$
    
    Here, α is the learning rate, and ϵ is a small constant to prevent division by zero.
    The below are variations of adam only…
    
2. AdaMax
3. Nadam
4. AdamW

---

**Learning Rate Scheduling**

starting with a large learning rate and then reducing it once training stops making fast progress is better than a constant learning rate

or start with a low learning rate, increase it, then drop it again. These strategies are called learning schedules.

1. Power scheduling
    
    $\alpha_t = \frac{\alpha_0}{(1 + kt)^\gamma}$
    
    - $\alpha_t$ is the learning rate at time step t.
    - $\alpha_0$ is the initial learning rate.
    - k is a hyperparameter that controls how quickly the learning rate decays.
    - γ is the power factor that determines the rate of decay (usually between 0 and 1).
    - t is the current time step (or epoch).

1. Piecewise constant scheduling
    - $\eta_0$ as the initial learning rate
    - $\eta_i$ as the learning rate at the i-th interval
    - $T_i$ as the epoch or iteration at which the learning rate changes
    
    $\eta(t) = \eta_i$ for $T_{i-1} < t < T_i$
    
2. Performance scheduling
    
    Define
    
    - η(t) as the learning rate at time t (epoch or iteration)
    - $\eta_{new}$ as the updated learning rate
    - ρ as the reduction factor (a value between 0 and 1).
    - metric(t) as the performance metric at time t (e.g., validation loss or accuracy).
    - patience as the number of epochs or iterations to wait before reducing the learning rate after the performance metric stops improving.
    - min_delta as the minimum change in the monitored metric to qualify as an improvement.
    
    Learning rate is updated as follows
    
    1. Initialize the learning rate to $\eta_0$.
    2. For each epoch or iteration t track the best performance metric observed so far
    if → metric(t) − best_metric > min_delta:
        1. Update best_metric to metric(t) 
        2. Reset epochs_since_improvement to 0.
        
        Else:
        
        1.  Increment epochs_since_improvement
        2. If epochs_since_improvement exceeds patience:
        3. Update the learning rate $\eta(t)$ to $\eta_{new} = \eta(t) \times \rho$
        4. Reset epochs_since_improvement to 0.

1. Exponential scheduling
    - $\eta_0$ as the initial learning rate.
    - γ as the decay rate, a constant between 0 and 1.
    - t as the current time step (epoch or iteration).
    
    Learning rate is given by 
    
    $\eta(t) = \eta_0 . \gamma^t$
    
2. 1cycle scheduling
    
    It follows a cyclical pattern with a single cycle, starting from an initial value, increasing to a maximum value, and then decreasing back to a minimum value.
    
    - $\eta_{min}$ as the initial minimum learning rate.
    - $\eta_{max}$ as the maximum learning rate.
    - T as the total number of iterations or epochs.
    - t as the current time step (iteration or epoch).
    - phase_1_end as the time step at the end of the first phase (halfway point).
    
    Learning rate at any time t is given by
    
    if  $t \le phase\_1\_end$  then $\eta_{min} + \frac{t}{phase\_1\_end} (\eta_{max} - \eta_{min})$
    
    if  $t \le phase\_1\_end$  then $\eta_{min} + \frac{t}{phase\_1\_end} (\eta_{max} - \eta_{min})$
    

---

### Regularization for Neural Networks

Models are prone to overfitting as there are a lot of parameters, regularization can be used to prevent this

Early stopping and batch normalization are already acting as regularizers.

1. L1 and L2 regularization
2. Dropout
    
    Working of Dropout
    
    At every training step, every neuron has a probability p of being temporarily “dropped out”, meaning it will be entirely ignored during this training step, but it may be active during the next step. The hyper parameter p is called the dropout rate, and it is typically set between 10% and 50%: closer to 20%–30% in recurrent neural nets, and closer to 40%–50% in convolutional neural networks.
    
    Neurons cannot co-adapt with neighboring neurons and they have to be as useful as possible on their own. Typically, dropout is turned off during inference, where all neurons are used to make predictions.
    
3. Monte Carlo (MC) Dropout
    
    **Training Phase**
    
    1. Apply dropout to the neural network as usual with a dropout rate p.
    2. Train the model on the training data with the standard optimization process.
    
    **Inference Phase with MC Dropout**
    
    1. Enable dropout during the inference phase (which is normally turned off).
    2. Perform multiple stochastic forward passes through the network for each input sample. Let’s say we perform N forward passes.
    3. Each forward pass results in a different set of neurons being dropped out, creating an ensemble of N different predictions for each input. After which the mean of the different results and the variance (uncertainty) can we checked which can be used for confidence assessment.
4. Max-Norm Regularization
    - W be the weight matrix of a particular layer.
    - ∥W∥ be the norm of the weight matrix.
    - c be the maximum allowed norm.
    
    Procedure
    
    1. Initialize the weights of the neural network.
    2. Define the maximum norm threshold c.
    3. During training, after each weight update, check the norm of the weight matrix
        
        If ∥W∥>c, rescale W to have a norm of c
        
        If the norm of the weight matrix exceeds the threshold cc, the weights are rescaled as follows:
        
        $W \leftarrow W . \frac{c}{|W|}$
        

---

## Custom Models and Training with TensorFlow

---

## Deep Computer Vision Using Convolutional Neural Networks

### **Step 1: Convolutional Layers**

Purpose: To extract features from the input image.

1. Filters/Kernels
    
    **Filters/Kernels**
    
     Small matrices (e.g., 3x3 or 5x5) that slide over the input image, computing dot products between the filter and patches of the input. This operation is called a convolution. 
    
    i/p → layer (filter/s) → o/p (feature map)
    
    The filters are not to be set manually but training the CNN will automatically learn the most useful filters for it’s task
    
2. Feature Maps
    
    The result of applying a filter to an input, highlighting specific features such as edges, textures, or patterns.
    
    Each convolutional layer outputs one feature map per filter.
    

Remember padding = “valid” is not setting zero-padding so the output feature map will reduce in size 

padding = “same” is adding zeros around the input image to ensure the output feature map has the same size as the input. Padding helps retain spatial dimensions after convolution.

The step size by which the filter moves over the input. A stride of 1 means the filter moves one pixel at a time.

### Step 2: Activation functions

simply Relu can you used to introduce non-linearity.

Why should i introduce non-linearity?

Linear models (without non-linear activation functions) can only learn linear relationships between inputs and outputs. This means they can only model straight-line relationships in the data. However, most real-world data exhibit complex, non-linear relationships. Non-linear activation functions allow the network to learn and approximate these complex patterns and functions.

### Step 3: Pooling layers

Reduces the spatial dimensions of the feature map, thereby reducing the number of parameters and computational load.

Max Pooling (selects the maximum value) and Average Pooling (computes the average value) → [i](http://i.ps)/ps not fulfilling the criteria are dropped

### Step 4: Fully **Connected Layer**

The output from the convolutional and pooling layers is flattened into a 1D vector and fed into one or more fully connected layers these are normal neural network layers.

**Data augmentation**

Data augmentation artificially increases the size of the training set by slightly shift, rotate, and resize every picture which in turn adds the resulting pictures to the training data reducing overfitting, making this a regularization technique

### Famous CNN architectures

1. **LeNET-5**
2. AlexNet
3. GoogleNet
4. VGGNet
5. ResNet
6. Xception
7. SENet

---

### Classification and Localization

This can be expressed as a regression task, the aim is to predict a bounding box around the image.

4 numbers are to be predicted → horizontal and vertical coordinates of the object’s center, as well as its height and width.

So to the same CNN model that were prepared above we just need to add a second dense output layer with four units typically on top of the global average pooling layer.

Intersection over Union (IoU) can be used instead of MSE as metric for evaluation

---

### Object detection

A sliding CNN is used to detect multiple objects

But this detects the same object multiple times so the unnecessary boxes can be eliminated using non-max suppression.

1. Fully Convolutional Network (FCN).
2. You Only Look Once (YOLO).

---

### Object tracking

1. DeepSORT

---

### Semantic segmentation

---

## Processing Sequences Using RNNs and CNNs

### Recurrent Neural Network

recognizes patterns in sequences of data, such as time series, speech, text, and video

**Recurrent layer**

At each time step t (also called a frame), the recurrent neuron receives the inputs $x_t$ as well as its own output from the previous time step, $\hat{y}_{t-1}$.

For a layer of recurrent neurons 

$\hat{y}_t = f(W_{x}.x_t + W_{\hat{y}}.\hat{y}_{t-1}+b)$

$W_x$ is the weight vector for inputs

$W_{\hat{y}}$ is the weight vector for output of the previous step t-1

**Input and output sequences**

sequence-to-sequence network: a sequence of inputs produces a sequence of outputs → daily power consumption

sequence-to-vector network: a sequence of inputs ignoring all outputs except for the last one → movie review

 vector-to-sequence network: the same input vector over and over again at each time step and let it output a sequence → image captioning

encoder (sequence-to-vector) → followed by → decoder (vector-to-sequence) → translation from one language to another

### Training RNNs

seasonality

trend

differencing

moving averages

### ARMA model family

It consists of two main types of models: the AR (AutoRegressive) model and the MA (Moving Average) model.

1.  **AutoRegressive (AR) Model**
    
    The AutoRegressive model specifies that the output variable depends linearly on its own previous values.
    
    $X_t = c + \phi_1 X_{t-1}+\phi_2 X_{t-2}+\cdot\cdot\cdot+\phi_p X_{t-p} + \epsilon_t$
    
    - $X_t$ is the time series at time t.
    - c is a constant.
    - ϕ1,ϕ2,…,ϕp are the parameters of the model.
    - $\epsilon_t$ is white noise error term at time t, typically assumed to be normally distributed with mean 0 and variance $\sigma^2$.
2. **Moving Average (MA) Model**
    
    The Moving Average model specifies that the output variable depends linearly on the current and past values of a stochastic (white noise) term.
    
    $X_t = c + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdot \cdot \cdot+ \theta_q\epsilon_{t-q}$
    
    - $X_t$ is the time series at time t.
    - c is a constant.
    - $\epsilon_t, \epsilon_{t-1},\cdot \cdot\cdot,\epsilon_{t-q}$ are white noise error terms.
    - $\theta_1, \theta_2, \cdot \cdot\cdot ,\theta_1$ are the parameters of the model.

Combining the AR and MA models we get

$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdot \cdot \cdot + \phi_p X_{t-p} + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdot \cdot \cdot+ \theta_q\epsilon_{t-q}$

- $X_t$ is the time series at time t.
- c is a constant.
- ϕ1,ϕ2,…,ϕp are the parameters of the AR part of the model.
- θ1,θ2,…,θq are the parameters of the MA part of the model.
- $\epsilon_t$ is white noise term.

**Differencing + ARMA = ARIMA (Auto Regressive Integrated Moving Average) model**

**Seasonality + ARIMA = SARIMA ( Seasonal Auto Regressive Integrated Moving Average) model**

 ****

### LSTM

special kind of RNN, capable of learning long-term dependencies

![Untitled](Machine%20learning%202025837b28ef4aecbc1b3261dbcfb669/Untitled%201.png)

Variants of LSTM

peephole

coupled forget and input gates

GRU

Excellent blog to understand LSTM : [[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)]

---

## Natural Language Processing with RNNs and Attention

---

## Autoencoders, GANs, and Diffusion Models

---

## Transformers

encoder [lstm] and decoder [lstm]

---

## Reinforcement Learning

---

## Graph Neural Networks (GNN)

---

### Application of graph machine learning

2 types of prediction task → graph-level, node-level, edge-level and community level

**Graph-level tasks**

Categorize graphs into types → discover new drugs

**Node-level tasks**

Classification of nodes → protein folding

**Edge-level tasks**

Missing link prediction → recommendation system like pinSage

**Community-level tasks**

Clustering task → traffic prediction

---

### **Graph Representation**

1. Directed vs undirected graph.
2. Weighted vs unweighted graph.
3. Connected vs unconnected graph.
4. Self loops and multi-graph.
5. Bipartite graph.
6. Adjacency matrix.
7. Edge list.
8. Adjacency list.

---

### Traditional methods for Machine learning in Graphs

given a graph extract node, link and graph level features and learn a model (SVM, neural network, etc) that maps features to labels

Input graph → Structured features → Learning algorithm → Prediction

Comes between Input graph and structured features → Feature engineering (node-level features, edge-level features, graph-level features)  

**Node-level features**

The following are node level features

**Node degree** 

**Node centrality**

1. **Eigen vector centrality**
    
    For a node v, it’s centrality is how many important neighbor node it has u $\in$  N(v), where N(v) are neighbors of v
    
    $c_v = \frac{1}{\lambda} \sum_{u \in N(v)} c_u$ 
    
    where $\lambda$ is some positive constant. 
    
2. **Betweenness centrality**
    
    If a node lies in many shortest paths between other nodes
    
    $c_v = \sum_{s \neq v \neq t} \frac{Number \ of \ shortest \ paths \ between \ s \ and \ t \ that \ contain \ v}{Number \ of \ paths \ between \ s \ and \ t}$
    
3. **Closeness centrality**
    
    smallest shortest path lengths to all other nodes
    
    $c_v = \frac{1}{\sum_{u \neq v} shortest \ path \ length \ between \ u \ and \ v}$
    

**Cluster coefficient**

measure how connected v’s neighboring nodes are

$e_v = \frac{Number \ of \ edges \ among \ neighboring\ nodes}{{k_v\choose 2}} \in [0,1]$

$k_v \choose 2$ is the number of node pairs among $k_v$ neighboring nodes

Measures how connected one’s neighboring nodes are → clustering coefficient

**Graphlets**

clustering coefficients counts the number of triangles in the ego network

Graphlets are rooted connected non-isomorphic subgraph

1. Graphlet degree vector
    
    Count vector  of graphlets rooted at a given node.
    
    where degree is the number of edges that a node touches and clustering coefficient counts the number of triangles that a node touches.
    
    GDV provides a measure of a node’s local network topology.
    

**Edge-level features**

For each pair (x,y) compute score c(x,y), sort these pairs in decreasing order and predict top n pairs as new links and see which of these links actually exist in the graph

Types of link level fetures

1. Distance based feature.
    
    shortest path distance between two nodes.
    
2. Local neighborhood overlap.
    
    Number of common neighbors between two nodes
    
    1. Common neighbors 
        
        $N(v_1) \cap N(v_2)$
        
    2. Jaccard’s coefficient
        
        $\frac{| N(v_1) \cap N(v_2)|
        }{|N(v_1) \cup N(v_2)|
        }$
        
    3. Adamic-Adar index
        
        $\sum_{v \in N(v_1) \cap N(v_2)} \frac{1}{log(k_u)}$ 
        
        here k is the degree
        
3. Global neighborhood overlap.
    
    Katz index.
    
    Count the number of paths of all lengths between a given pair of nodes.
    
    Computing the number of paths between two nodes = computing the powers of graph’s adjacency matrix
    
    $P_{uv}^{(k)} =$ Number of paths of length k between u and v 
    
    $P^{(k)} = A^k$
    
    here $P^{(1)}_{uv}$ is the number of paths of length 1 between u and v in Adjacency matrix
    
    so katz index between $v_1$ and $v_2$ is calculated as sum over all path lengths
    
    $S_{v_1v_2} = \sum_{l=1}^{\infty} \beta^l A_{v_1v_2}^l$
    
    where 
    
    0 < $\beta$ < 1 is the discount factor
    
    $A_{v_1v_2}^l$ is the number of paths of length l between $v_1$and $v_2$.
    

**Graph-level features**

Graph kernel and types

kernel K($G, G') \in \mathbb{R}$  measure similarity between data points.

Bag-of-Words (BoW) → key idea → bag-of-*

1. Graphlet kernel
    
    count the number of different graphlets in the graph
    
    here graphlets are not rooted and don’t have to be connected → different from node level graphlets (check standford for clarification)
    
    $K(G, G') = f_G^T f_{G'}$
    
    However if G and G’ have different sizes we have to normalize the feature vector
    
    $K(G, G') = h_G^T h_{G'}$ where $h_G = \frac{f_G}{Sum(f_G)}$
    
    However this can be expensive and is NP-Hard
    
2. Weisfeiler-Lehman kernel
    
    Using neighborhood structure
    
    Color refinement
    
    A graph G with node V
    
    Give a color $c^{(0)}(v)$ to each node v
    
    Then refine node colors by
    
    $c^{(k+1)}(v) = HASH(\{c^{(k)}(v), \{ c^{(k)}(u)\}_{u \in N(v)}\})$    → The refined node color is a hash value of existing node color and color from it’s neighbors
    
    k is the number of steps → K-hop neighborhood
    
    After color refinement WL kernel counts nodes with given color
    
    $K(G, G') = \phi(WLK)^T \phi(WLK')^T$ = 49 → or any other value this is the inner product of the color count vectors
    
    WLK is computationally efficient
    
3. Random walk kernel
4. shortest path graph kernel

---

**Node Embeddings** 

Node → represent in the form of a vector of dimension d also called embedding

encode(nodes) → embedding space

encode(u) → $Z_u$

encode(v) → $Z_v$

Similarity(u, v) = $z_v^T z_u$ (similarity of embedding) →similarity of nodes correspond to the similarity of embedding

encoding approaches

Shallow encoding →  encoding is just an embedding-lookup → each node is assigned a unique embedding vector

1. DeepWalk
2. node2vec

decoder → maps from embedding to similarity score

maximize $z_v^T z_u$ for node pairs(u,v) that are similar

---

**Random walk approaches for Node embedding**

random walks are expressive (incorporates lower and higher order neighborhood info) and efficient (as only a pair of nodes is considered not whole graph)

what to do? → find embedding of a node in d-dimensional space

Idea → learn node embedding in a way such that nearby nodes are closer in the network

How to define nearby nodes? → $N_R(u)$ neighborhood of u is defined by some random walk strategy R

Find vector $z_u$ (embedding) given node u.

Probability P(v | $z_u$) → predicted probability of visiting node v on random walks starting from node u.

~

To find the prediction probability we use non-linear function

1. Softmax function
    
    turns vector of K real values (model prediction) into K probabilities that sum to 1
    
    $\sigma(z) = \frac{e^{z_i}}{\sum_{j=1}^Ke^{z_j}}$
    
2. Sigmoid function
    
    S-shaped function that turns real values between 0 and 1
    
    $S(x) = \frac{1}{1+e^{-x}}$
    

Random walk

Sequence of points visited in a random way is called random walk.

Random walk embedding

$z_v^T z_u$ ⇒ probability that u and v co-occur on a random walk over the graph

1. Estimate probability of visiting node v on a random walk starting from node u using some random walk strategy R.
2. Optimize embedding to encode these random walk statistics.

**Feature learning optimization**

G = (V, E)

our goal is to learn a mapping f: u → $\mathbb{R}^d:$

f(u) = $z_u$

log likelihood objective:

$\max_{f} \sum_{u \in V} logP(N_R(u) | z_u)$ → find a function such that for all the nodes in the graph it maximizes the log probability of the nodes in the neighborhood.

where $N_R(u)$ is the neighborhood of node u by strategy R

1. Run short fixed-length random walks starting from each node u in the graph using some random walk strategy R.
2. For each node u collect $N_R(u)$, the multiset* of nodes visited on random walks starting from u.
3. Optimize embedding according to: Given node u, predict its neighbors $N_R(u)$ → using SGD
    
    $\max_{f} \sum_{u \in V} logP(N_R(u) | z_u)$   → maximum likelihood objective
    
    given the embedding of node u the probability of multi-set $N_R(u)$ is maximized.
    
    This can also be written as
    
    $L = \sum_{u \in V} \sum_{v \in N_R(u)} -log(P(v|z_u))$
    
    $P(v|z_u) = \frac{exp(z_u^T.z_v)}{\sum_{n \in V}exp(z_u^T.z_{v'})}$ → Why softmax? → node v should be similar to node u out of all nodes n
    
    This requires each nodes dot product with every other node → which is computationally expensive
    
    Find embeddings $z_u$ that minimize L
    
    $\sum_{n \in V}exp(z_u^T.z_{v'})$ → this normalization is very expensive
    
    to tackle this we use negative sampling
    
    Rather than summing over all the nodes we sum over only a few node that are chosen using negative sampling
    
    $\frac{exp(z_u^T.z_v)}{\sum_{n \in V}exp(z_u^T.z_{v'})} \approx log(\sigma(z_u^T.z_v)) - \sum_{i=1}^{k}log(\sigma(z_u^T.z_{n_{i}})), n_i \sim P_V$
    
    here k is the number of negative samples
    
    How to select negative samples?
    
    Probability of choosing a node is proportional to its degree
    
    1. higher k gives more robust estimates
    2. higher k values correspond to higher bias
    
    in practice k = 5 to 20
    

We use SGD to optimize the objective function L

$L = \sum_{u \in V} \sum_{v \in N_R(u)} -log(P(v|z_u))$

1. Initialize $z_i$ at some randomized value for all i.
2. Iterate until convergence:   $L^{(u)} = \sum_{v \in N_R(u)} -log(P(v|z_u))$
    1. For all i, compute the derivative $\frac{\delta L
    }{\delta z_i}$
    2. for all l, make a step towards the direction of derivative: $z_i$ ← $z_i - \eta \frac{\delta L}{\delta z_i}$
    

---

**node2vec**

like skip-gram model → predict context words → in graph → predict context nodes

skip-gram with negative sampling

random walks are used to generate context nodes

second-order graph traversal

Random Walk Generation

1. Graph Representation
    
    Consider a graph G=(V,E) with V as the set of nodes and E as the set of edges.
    
2. Random Walk Strategy
    
    Node2Vec introduces two parameters p and q, to control the walk behavior:
    
    p: Return parameter, controlling the likelihood of revisiting the previous node.
    
    q: In-out parameter (inout parameter), controlling the likelihood of visiting nodes further away.
    
3. Transition Probabilities
For a walk starting at node t and currently at node v, the next step to node x is determined by the transition probability:
    
    $\pi_{vx} = \alpha_{pq}(t,x).w_{vx}$
    
    where $w_{vx}$ is the weight of the edge between v and x, and $\alpha_{pq}(t, x)$ is a bias factor based on p and q.
    
4. Bias Factor Calculation
    
    $\alpha_{pq}(t, x)$ is defined as:
    
    $\alpha_{pq}(t, x) = 
    \begin{cases}
          \frac{1}{p} \ if \ d_{tx} = 0\\
          1 \ if \ d_{tx} = 1 \\ 
          \frac{1}{q} \ if \ d_{tx} = 2 \\
        \end{cases}$  
    
    where $d_{tx}$ is the shortest path distance between t and x
    
5. Generating Walks
For each node u in the graph, generate multiple random walks of fixed length. The walks are influenced by p and q, allowing for a balance between depth-first and breadth-first search strategies.

---

**Embedding entire graph**

Approach 1

embedding of graph = sum of embedding of all nodes in the graph

Approach 2

introduce a subgraph in the graph and find embedding for this sub-graph

Approach 3

anonymous walk embedding

An anonymous walk of length k in a graph is a sequence of integers ($a_1, a_2, ... , a_{k+1}$) where each integer represents the position or role of a node in the walk, not its specific identity. 

Example of Anonymous Walk

Consider a graph with nodes A,B,C,D and a random walk starting from node A with the sequence A→B→C→A→D .

1. Original Random Walk:
    
    sequence A→B→C→A→D 
    
2. Anonymous Walk:
3. Position indices: (1,2,3,1,4)
    
    Explanation:
    
    - A is visited first (index 1).
    - B is visited second (index 2).
    - C is visited third (index 3).
    - A is revisited, so it retains index 1.
    - D is visited for the first time after the revisits, so it gets index 4.

The number of anonymous walks grows exponentially

How to use anonymous walks?

simulate anonymous walks $w_i$ of $l$ steps and record their counts

Represent the graph as a probability distribution over these walks

example -

set $l = 3$

Then we can represent the graph as a 5-dimensional vector

- since there are 5 anonymous walks $w_i$ of length 3:111, 112, 121, 122, 123

$z_G[i] =$ probability of anonymous walk $w_i$ in G

How many random walks do we need?

we want the distribution to have error of more than $\epsilon$ with probability less than $\delta$:

$m = [ \frac{2}{\epsilon^2} (log(2^\eta - 2) - log(\delta) ) ]$

where $\eta$ is the number of anonymous walks of length $l$

For example: 

There are $\eta$=877 anonymous walks of length 7. If we set $\epsilon$=0.1 and $\delta=0.01$ then we need to generate m=122,500 random walks

Approach 4

Learn graph embedding $Z_G$ together with all the anonymous walk embedding $z_i$ of anonymous walk $w_i$

Z = {$z_i:i=1 ... \eta$}, where $\eta$ is the number of sampled walks

learn vector $Z_G$ for input graph

starting from node 1 sample anonymous random walks

learn to predict walks that co-occur in $\Delta$-size window (eg. predict $w_2$ given $w_1,w_3$ if $\Delta=1$

objective:

$max \sum_{t = \Delta}^{T-\Delta}logP(w_t|w_{t-\Delta},...,w_{t-\Delta},z_G)$

sum the objective over all nodes in the graph

Once you learn $z_G$ we can use it for graph classification, etc

---

### Graph as a matrix: Pagerank, Random walks and embeddings

**Link Analysis Algorithms**

1. **Pagerank**
    - Measure the importance of nodes in a graph using link structure of the web.
    - Models a random web surfer using the stochastic adjacency matrix M.
    - Pagerank solves r = M.r    where r can be viewed as both the principle eigenvector of M and as the stationary distribution of random walk over the graph.
    
    There are 2 types of links → in-links and out-links 
    
    in-links are more important as they are hard to control
    
    Pagerank
    
    The vote from important page is worth more:
    
    - Each link’s vote is proportional to the importance of its source page
    - if page i with importance $r_i$ has $d_i$ out-links, each link gets $r_i/d_i$ votes
    - page j’s own importance $r_j$ is the sum of the votes on it’s in-links
    
    $r_j = \sum_{i \rightarrow j} \frac{r_i}{d_i}$    ——— > equation form
    
    here $d_i$ is the out-degree of node i
    
    This will give you “Flow” equations:
    
    $r_y = r_y/2 + r_a/2$
    
    $r_a = r_y/2 + r_m$
    
    $r_m = r_a/2$
    
    How to solve them → we can use gaussian elimination but that is expensive
    
    Lets represent the graph as a matrix
    
    Stochastic adjacency matrix M  → this is just the adjacency matrix
    
    let page j have $d_j$ out-links
    
    if j → i, then $M_{ij}$ = $\frac{1}{d_j}$                 —————— > here the out-links are divided over 1 (all columns should sum to 1)
    
    M is a column stochastic matrix meaning columns sum to 1
    
    Rank vector r: an entry per page
    
    $r_i$ is the importance score of page i
    
    $\sum_i r_i = 1$     →   the entries of vector r (all the pages) has to sum to 1  →  probability distribution over the nodes in the network
    
    The flow equation can be written as
    
    $r = M.r$             —— >  in the matrix forum
    
    $r_j = \sum_{i \rightarrow j}\frac{r_i}{d_j}$    ——— > equation form
    
    ![graph.png](Machine%20learning%202025837b28ef4aecbc1b3261dbcfb669/graph.png)
    
    Equation form
    
    $r_y = r_y/2 + r_a/2$
    
    $r_a = r_y/2 + r_m$
    
    $r_m = r_a/2$
    
    Matrix form
    
    |  | $r_y$ | $r_a$ | $r_m$ |
    | --- | --- | --- | --- |
    | $r_y$ | 1/2 | 1/2 | 0 |
    | $r_a$ | 1/2 | 0 | 1 |
    | $r_m$ | 0 | 1/2 | 0 |
    
    $\begin{bmatrix}
    r_y \\ 
    r_a \\
    r_m
    \end{bmatrix}  = 
    \begin{bmatrix}
    1/2 & 1/2 & 0 \\ 
    1/2 & 0 & 1 \\
    0 & 1/2 & 0
    \end{bmatrix}  
    \begin{bmatrix}
    r_y \\ 
    r_a \\
    r_m
    \end{bmatrix}$   
    
    $r$                                  $M$                         $r$
    
    p(t) vector whose ith coordinate is the probability that the surfer is at page i  at time t
    
    so, p(t) is the probability distibution over pages
    
    Where is the surfer at time t+1
    
    ![Untitled](Machine%20learning%202025837b28ef4aecbc1b3261dbcfb669/Untitled%202.png)
    
    pick any outlink at uniform and go to it
    
    p(t+1) = M. p(t)
    
    after some time a random walk reaches a state 
    
    p(t+1) = M .p(t) = p(t)      …. no change p(t) is a stationary distribution of a random walk
    
    our original rank vector r satisfies r = M.r
    
    so the flow equation 
    
    1.r = M.r
    
    so the rank vector r is a eigenvector of stochastic adj matrix M
    
    How to solve pagerank?
    
    - Assign each node an initial page rank.
    - Repeat until convergence      $( \sum_i|r_i^{t+1} - r_i^t| < \epsilon)$
        
        calculate the page rank of each node
        
        $r_j^{(t+1)} = \sum_{i \rightarrow j} \frac{r_i^{(t)}}{d_i}$
        
    
    Solve using Power iteration method
    
    - Initialize $r^0 = [1/N,....,1/N]^T$
    - Iterate $r^{(t+1)} = M.r^t$
    - stop when $|r^{t+1} - r^t|_1 < \epsilon$
    
    about 50 iterations is sufficient to estimate limiting solution
    
    Google is computing this pagerank everyday over the entire web-graph(10 billion+ nodes)
    
    Problems
    
    1. Dead ends
        
        have no out-links
        
        solution → teleport out with probability 1.0
        
    2. Spider traps
        
        All out-links are within the group
        
        solution → teleport out of the spider trap within a few time steps with probability < 1.0
        
    
    So the pagerank equation becomes
    
    $r_j = \sum_{i \rightarrow j} \beta \frac{r_i}{d_j} + (1 - \beta) \frac{1}{N}$   
    
    In matrix form
    
    P = $\beta$ M + ($1 - \beta)[\frac{1}{N}]_{N \times N}$
    
2. Personalized Pagerank (PPR)
    
    Only difference is that instead of teleporting to any node in a graph we only teleport to a subset of the nodes in the graph
    
3. Random walk with restarts
    
    Teleport always to the starting node.
    

### Introduction to graph neural Networks

Formulating the task as an optimization problem

$min_\theta L(y, f(x))$ ← is the objective function

here $\theta$ is the set of features we optimize

$\theta$ = {Z} could contain one or more scalars, vectors or matrices

One common loss for classification is the cross entropy (CE) 

How to optimize the objective function

Gradient Descent

Stochastic gradient descent

Mini-batch gradient descent

### A GNN layer

gnn layer = message + aggregation

1. How we define a gnn layer?
2. How we stack those gnn layer?
3. How we create a computation graph?

**A Single Layer of GNN**

message + aggregation → GNN layer

{ GCN, GraphSage, GAT → different gnn architectures}

![Untitled](Machine%20learning%202025837b28ef4aecbc1b3261dbcfb669/Untitled%203.png)

1. Message Computation
    
    $m^{(l)} = MSG^{(l)}(h_u^{l-1})$
    
    $m^{(l)} = W^{(l)}(h_u^{l-1})$
    
    Multiply node features with Weight matrix $W^{(l)}$
    
2. Aggregation
    
    $h_v^{(l)} = AGG^{(l)}(\{ m_u^{(l)}, u \in N(v) \})$
    
    AGG function could be anything eg:- sum(), mean() or max() aggregator
    
    $h_v^{(l)} = SUM^{(l)}(\{ m_u^{(l)}, u \in N(v) \})$
    

issue → Information from node itself gets lost

solution → Include prev layer embedding of node v when computing for new layer

So it becomes

1. Message Computation
    
    $m_u^{(l)} = W^{(l)}(h_u^{l-1})$                    For neighborhood nodes u 
    
    $m_v^{(l)} = B^{(l)}(h_v^{l-1})$                      For node v itself
    
2. Aggregation
    
    $h_v^{(l)} = CONCAT(AGG^{(l)}(\{ m_u^{(l)}, u \in N(v) \}), m_v^{(l)})$
    
    you can use concatenation or summation
    
3. Non-linear activation
    
    Adds expressiveness
    
    $\sigma(.), Relu(.) , etc$
    

These three steps become a single layer of GNN

### Classical GNN layer types

1. Graph Convolutional Network (GCN)
    
    $h_v^{(l)} = \sigma(\sum_{u\in N(v)} W^{l}\frac{h_u^{(l-1)}}{|N(v)|})$
    
    1. Message computation
        
        $m_u^{(l)} = \frac{1}{N(v)} W^{(l)}(h_u^{l-1})$                ← Normalized the degree of the node
        
    2. Aggregation
        
        $h_v^{(l)} = \sigma(sum^{(l)}(\{ m_u^{(l)}, u \in N(v) \}))$
        
    
    In matrix form
    
    1. $X^{(k+1)} = \sigma(D^{-1}AX^{(k)}W^{(k)})$
    2. $X^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})$
    
    D is degree matrix (diagonal matrix)
    
    A is adjacency matrix
    
    X is feature or embedding matrix
    
    W is weight matrix
    
    D inverse is calculate to ensure normalization so that nodes with higher degrees do not disproportionately affect the aggregation or propagation process
    
    Both formulas are the same only in second formula the degree matrix is normalized on both size of adjacency matrix to ensure numerical stability
    
2. GraphSage
    
    $h_v^{(l)} = \sigma(W^{(l)}. CONCAT(h_v^{l-1},AGG(\{ h_u^{(l-1)}, \forall u \in N(v)  \})))$
    
    aggregation of a node’s self embedding along with neighbors
    
    1. Message computation
        
        $AGG(\{ h_u^{(l-1)}, \forall u \in N(v)  \})$
        
    2. Aggregation
        1. Stage 1
            
            $h_{N(v)}^{(l)} \leftarrow AGG(\{ h_u^{(l-1)}, \forall u \in N(v)  \})$
            
        2. Stage 2
            
            $h_v^{(l)} \leftarrow \sigma(W^{(l)}. CONCAT(h_v^{l-1}, h_{N(v)}^{(l)}))$
            
    
    Aggregation functions
    
    1. Mean
        
        $AGG = \sum_{u \in N(v)} \frac{h_u^{(l-1)}}{|N(v)|}$     
        
    2. Pool (mean or max any pooling can be used)
        
        $AGG = Mean(\{ MLP(h_u^{(l-1)}), \forall u \in N(v) \})$
        
    3. LSTM
        
        $AGG = LSTM([h_u^{(l-1)},\forall u \in \pi (N(v))])$
        
        here LSTM is a sequence model that ignores the order of the nodes
        
    
    As embedding vectors have different scales
    After applying L2 normalization all vectors will have same scales
    
3. The Label Propagation Algorithm (LPA)
    
    $Y^{(k+1)} = D^{-1}AY^{(k)}$                       
    Here all the nodes propagate label to their neighbors
    
    $y_i^{(k+1)} = y_i^{(0)}, \forall i \le m$                 
    
    Here all labeled nodes are reset to their initial values as we want to persist the labels of already labelled nodes
    
4. GCN-LPA
    
    increasing the strength of edges between the nodes of the same class is equivalent to increasing the accuracy of LPA’s predictions
    
    We want to find optimal node embeddings W* and edge weights A* by setting 
    
    $W^* A^* = arg min_{W,A} L_{gcn}(W,A) + \lambda L_{gcn} (A)$
    
5. GAT
    
    $h_v^{(l)} = \sigma(\sum_{u \in N(v) }\alpha_{vu} W^{(l)}h_{(u)}^{(l-1)})$
    
    here $\alpha_{vu}$ are the attention weights
    
    In GCN/GraphSage $\alpha_{vu}$ =$\frac{1}{|N(v)|}$  making all neighbors equally important
    
    $\alpha_{vu}$ is the importance of information coming for node u for the node v → giving each neighbor different neighbors
    
    $\alpha_{vu}$ focuses on the important part of the data and ignores the rest
    
    for attention we compute attention coefficients $e_{vu}$
    
    $e_{vu} = a(W^{(l)}h_{u}^{(l-1)},W^{(l)}h_{v}^{(l-1)})$
    
    $e_{vu}$ indicates the importance of u’s mesage to node v
    
    Normalize $e_{vu}$ into final attention weights $\alpha_{vu}$
    
    Using the softmax function so that $\sum_{u\in N(v)} \alpha_{vu} = 1$
    
    $\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k\in N(v)}\exp(e_{vk})}$
    
    Weighted sum based on final attention weight $\alpha_{vu}$
    
    $h_v^{(l)} = \sigma(\sum_{u \in N(v) }\alpha_{vu} W^{(l)}h_{(u)}^{(l-1)})$
    
    What is the form of the attention mechanism (a)
    
    $e_{vu} = a(W^{(l)}h_{u}^{(l-1)},W^{(l)}h_{v}^{(l-1)})$
    
    here a could be a single-layer neural network
    
    $e_{vu} = Linear(Concat(W^{(l)}h_{u}^{(l-1)},W^{(l)}h_{v}^{(l-1)}))$
    
    Multi-headed attention
    
    stabilizes the learning process of attention mechanism
    
    create multiple attention scores[each replica has different set of parameters]:
    
    $h_v^{(l)}[1] = \sigma(\sum_{u \in N(v) }\alpha_{vu}^{1} W^{(l)}h_{(u)}^{(l-1)})$
    
    $h_v^{(l)}[2] = \sigma(\sum_{u \in N(v) }\alpha_{vu}^{2} W^{(l)}h_{(u)}^{(l-1)})$
    
    $h_v^{(l)}[3] = \sigma(\sum_{u \in N(v) }\alpha_{vu}^{3} W^{(l)}h_{(u)}^{(l-1)})$
    
    Outputs are aggregated
    
    $h_v^{(l)} = AGG(h_v^{(l)}[1],h_v^{(2)}[2],h_v^{(3)}[3])$
    
    AGG could be concatenation or summation
    

General GNN layer 

- Linear
- BatchNorm
- Dropout
- Activation
- Attention
- Aggregation

### Things to work on graphs

Class imbalance

Dataloader

Data transformation

Models

GCN

GraphSage

GAT

GATv2Conv

GCN-LPA

Visualisation?

Types of graphs → directed, weighted?

Why and how to use graph models

- **Integration**: Node2Vec embeddings can serve as initial node representations fed into GNNs, enhancing the GNN's ability to generalize and capture nuanced graph structures.
- **Enhancement**: GNNs can further refine node embeddings learned by Node2Vec through iterative message passing and feature aggregation, improving their robustness and discriminative power for specific tasks.

List of Graph modesl

1. Embedding based models
    1. DeepWalk
    2. node2vec
    3. LINE
2. Convolution graph models
    1. GCN
    2. GraphSAGE
    3. GAT
    4. ChebNet
    5. MixHop
    6. Graph Isomopphism Network (GIN)
    7. Graph Convolution Matrix Completion (GCMC)
3. Recurrent and other model
    1. Gated Graph Neural Networks (GGNN)
    2. Graph Recurrent Neural Network (GRNN)
    3. Graph U-Net
4. Graph Generative Models
    1. GraphRNN
    2. GraphVAE
5. Graph Reinforcement Learning Models
    1. DQN-GNN
    2. Graph Actor-Critic

---

Built using neural network

### Natural Language Processing (NLP)

1. **BERT (Bidirectional Encoder Representations from Transformers)**: A transformer-based model designed for understanding the context of words bidirectionally.
2. **GPT (Generative Pre-trained Transformer)**: An autoregressive transformer model known for generating coherent and contextually relevant text.
3. **T5 (Text-To-Text Transfer Transformer)**: A model that frames all NLP tasks as text-to-text problems.
4. **RoBERTa (Robustly Optimized BERT Pretraining Approach)**: An optimized version of BERT with more training data and longer training times.
5. **XLNet**: An autoregressive model that incorporates permutation-based training to capture bidirectional context.
6. **ERNIE (Enhanced Representation through Knowledge Integration)**: A model incorporating external knowledge graphs into the pre-training process.
7. **Transformer-XL**: A model addressing long-context dependencies by introducing segment-level recurrence mechanisms.
8. **DistilBERT**: A smaller, faster, cheaper, and lighter version of BERT.

### Computer Vision

1. **AlexNet**: An early convolutional neural network (CNN) that won the ImageNet competition in 2012.
2. **VGGNet**: Known for its simplicity and use of very small (3x3) convolution filters.
3. **GoogLeNet (Inception)**: Introduced the Inception module, allowing for more efficient computation.
4. **ResNet (Residual Networks)**: Introduced residual learning to train very deep networks.
5. **DenseNet**: Features dense connections between layers, encouraging feature reuse.
6. **MobileNet**: Designed for efficient use on mobile and embedded vision applications.
7. **EfficientNet**: Balances network depth, width, and resolution to improve performance.
8. **YOLO (You Only Look Once)**: Real-time object detection system.
9. **RCNN (Region-based Convolutional Neural Networks)**: Object detection model using region proposals.

### Reinforcement Learning

1. **DQN (Deep Q-Network)**: Combines Q-learning with deep neural networks to play Atari games at superhuman levels.
2. **A3C (Asynchronous Advantage Actor-Critic)**: Improves the stability and efficiency of training reinforcement learning models.
3. **AlphaGo**: The first AI to defeat a professional human player in the game of Go.
4. **AlphaZero**: Generalized version of AlphaGo that mastered Go, Chess, and Shogi from self-play.
5. **OpenAI Five**: AI system that defeated professional Dota 2 players.

### Speech Recognition and Generation

1. **DeepSpeech**: An end-to-end deep learning-based speech recognition system.
2. **WaveNet**: A generative model for raw audio waveforms, producing high-fidelity speech synthesis.
3. **Tacotron**: Text-to-speech model converting text to spectrograms, later used by WaveNet to produce audio.

### Generative Models

1. **GAN (Generative Adversarial Networks)**: A framework where two neural networks (generator and discriminator) are trained simultaneously to generate realistic data.
2. **DCGAN (Deep Convolutional GAN)**: Applies convolutional networks in GANs, particularly effective for image generation.
3. **StyleGAN**: Generates high-quality images with adjustable style parameters.
4. **VAE (Variational Autoencoder)**: A type of autoencoder that generates new data points by learning the probability distribution of input data.

### Sequence Models

1. **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) designed to handle long-term dependencies.
2. **GRU (Gated Recurrent Unit)**: A simplified version of LSTM with fewer parameters.
3. **Seq2Seq**: Encoder-decoder model for tasks like machine translation.

### Multimodal Models

1. **CLIP (Contrastive Language–Image Pretraining)**: A model trained on a variety of images paired with their descriptions, enabling it to understand images and text together.
2. **DALL-E**: A model capable of generating images from textual descriptions.

Reinforcement Learning

---

[Numpy](https://www.notion.so/Numpy-b6a09816adcb4f82be40ca89816c09db?pvs=21)

At each time step t (also called a frame), this recurrent
neuron receives the inputs x as well as its own output from the previous
time step, ŷ .