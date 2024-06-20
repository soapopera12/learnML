# Machine learning

**Notations** 

Important - purple

questions - red

Links - green

This is the link to Excalidraw

[https://excalidraw.com/#json=i1UW2iD_kiEFwWCRqL5Uq,b1nCADVf58OCrjqzk9GHGg](https://excalidraw.com/#json=i1UW2iD_kiEFwWCRqL5Uq,b1nCADVf58OCrjqzk9GHGg)

What is a derivative?

A derivative of a function represents the rate at which a function's value changes as its input changes. In simpler terms, the derivative measures how a function's output changes in response to changes in its input.

[https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3](https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3)
ś
What is a limit of a function?

It describes the behavior of a function as its input approaches a certain value

[https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3](https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3)

What are the rules of differentiation?

[https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3](https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3)

**Loss Types**

**MSE / L2**

**MAE / L1**

**RMSE**

**Huber loss = MAE + MSE**

**Binary Cross-Entropy Loss / Log Loss** 

only for dual classes

**Sparse categorical cross entropy** 

When you have a 10 possibilities and only one of them should be close to 1 eg: [ 0 0 0 0 1 0 0 0 0 0] → like for digit recognization

---

## Supervised Learning Techniques

---

### ***Linear regression***

weighted sum of input features. | hypothesis function

Loss function - MSE | RMSE

How to get values of the coefficients (weights)

**Normal equation**

**SVD**

**Gradient Descent**

1. start by filling θ with random values (this is called random initialization).
2. Compute the Cost Function (MSE).
3. Compute the Gradient (derivative of the cost function that is being used).
4. Update the Parameters.
5. Repeat Until Convergence.

The derivative of the cost function with respect to each parameter tells us how the cost function changes as we change each parameter. Specifically, it indicates the direction in which we should adjust the parameters to decrease the cost function.

MSE cost function is always convex → meaning no local minima just a single global minimum

cost function → shape of bowl (elongated (no scaling) or otherwise simple (scaling))

Types of gradient descent

Batch gradient Descent → for full dataset

Stochastic gradient descent → for single random instance

mini-batch gradient descent → for small parts of dataset

**Regularization of linear models**

reduces overfitting.

Must scale the data before regularization as it is sensitive to scale of input features.

Basically the value of MSE increase by adding a certain regularization parameter to the MSE. This will also increase the value of derivation of MSE, which should reduce the weights of the parameters by a slightly larger value during the update parameters step when compared to when we were not using regularization. Thereby keeping the weights as small as possible.

How does L1 and L2 reduce the different weights present int the model?

[https://www.quora.com/How-does-the-L1-regularization-method-help-in-feature-selection](https://www.quora.com/How-does-the-L1-regularization-method-help-in-feature-selection)

To understand how L1 helps in feature selection, you should consider it in comparison with L2.

- L1’s penalty: Σwi
- L2’s penalty: Σ(wi)2

Observation:
 L1 penalizes weights equally regardless of the magnitude of those weights. L2 penalizes bigger weights more than smaller weights.

For example, suppose w3=100 and w4=10

- By reducing w3 by 1, L1’s penalty is reduced by 1. By reducing w4 by 1, L1’s penalty is also reduced by 1.
- By reducing w3 by 1, L2’s penalty is reduced by 199. By reducing w_4 by 1, L2’s penalty is reduced by only 19. Thus, L2 tends to prefer reducing w3 over w4.

In general, when a weight wi has already been small in magnitude, L2 does not care to reduce it to zero, L2 would rather reduce big weights than eliminate small weights to
0. The result is that the weights are reduced, but almost never reduced to 0, i.e. almost never be completely eliminated, meaning no feature selection. On  the other hand, L1 cares about reducing big weights and small weights equally. For L1, the less informative features get reduced. Some features may get completely eliminated by L1, thus we have feature selection.

Tl;DR

Increasing the MSE will also increase the weights to be decreased at each instance of gradient descent.

Ridge regression (L2)

Lasso regression (L1)

Elastic Net regression (Both)

**Early stopping**

---

### **Polynomial regression**

---

### Classification

linear classification

non-linear classification

1. If some dataset is not linearly separable you can always add polynomial features resulting in linearly separable dataset.
2. Add similar computed features.

### **Logistic regression**

weighted sum of input features (same as regression) | hypothesis function | sigmoid function (output of the weighted sum of input features is passed to this function)

How Sigmoid function squeezes the input values between 0 and 1?

$Sigmoid(x) = \frac{1}{1 + e^{-x}}$

1. x can be a positive or a negative number.
2. if x is very large $e^{-x}$ becomes very small (close to 0), and 1 + 0 becomes 1 so the reciprocal of 1 is 1.
3. If x is very small $e^{-x}$ becomes very large so the sum $1 + e^{-x}$ becomes very large and the reciprocal is close to 0.
4. Therefore for a large value of x the function becomes 1 and for negative value the function becomes 0, the function output is 0.5 for value of x around 0.

cost function

$LOSS = -\frac{1}{N}\sum_{i=1}^N [y_i log(\hat{y_i}) + (1 - y_i) log(1 - \hat{y_i})]$

$\hat{y_i}$ is predicted value (0 or 1).

$y_i$ is actual value (0 or 1).

If $\hat{y_i}$  is close to 1 $log(\hat{y_i})$ is close to 0.

If $\hat{y_i}$  is close to 0 $log(1 - \hat{y_i})$ will be close to 1.

- If $y_i$ is 1 and $\hat{y_i}$  is close to 1,  $log(\hat{y_i})$ is close to 0, resulting in a low loss.
- If $y_i$ is 1 and $\hat{y_i}$  is close to 0,  $log(\hat{y_i})$ is very negative, resulting in a high loss.
- If $y_i$ is 0 and $\hat{y_i}$  is close to 0,  $log(\hat{y_i})$ is close to 0, resulting in a low loss.
- If $y_i$ is 0 and $\hat{y_i}$  is close to 1,  $log(\hat{y_i})$ is very negative, resulting in a high loss.

Finally Averaging the loss across the whole dataset gives an estimate of how well is our model doing.

Evaluation of Logistic regression

True Positive (TP), True Negative (TN), False Positive (FP) and False Negative (FN)

Accuracy = $\frac{TP + TN}{TP+TN+FP+FN}$ (how often are models prediction and correct)

Precision = $\frac{TP}{TP + FP}$ (how many positive predictions are actually correct)

Recall = $\frac{TP}{TP + FN}$ (how many actual positives were predicted correctly)

F1 score = $2 \times \frac{Precision \times Recall}{Precision + Recall}$ (harmonic mean of precision and recall)

AUC-ROC → True positive rate (Recall) vs False positive rate (should be close to 1)

AUC-PR → Precision (y-axis) vs recall (x-axis) curve (should be close to 1)

**Softmax regression**

support multiple classes

---

### Support Vector Machine (SVM)

widest possible street between classes - basically the best hyperplace that is being used to separate the instances of the two classes (line → plane → hyperplane)

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

creates a flowchart like structure where each node is based on a decision based on the value of the feature - significant features are chosen based on Gini impurity and Information gain

so branches represent decision and leaves represent final output or classification outcome.

feature scaling is not required mostly

Gini impurity

each node has a gini (impurity means how instances dont follow a particular nodes rule)

Entropy (Information gain)

how much information is given by a feature about a particular class

---

### Ensemble models and random forest

wisdom of the crowd

aggregating prediction of a group of predictors

group of decision trees on different smaller subsets of data

such ensemble of decision trees is called → Random forest

Ensemble methods are as follows

voting classifier

class that gets the most votes across wide variety of predictors is called hard voting classifier

diverse set of classifiers should be used

soft voting - average of all individual classifiers

Bagging and pasting

same training algo but different random subsets of training algo for each predictor

bagging - sampling with replacement (bootstrap aggregating - full form)

after a datapoint is chosen to be a part of the sample it is replaced back into the dataset to be picked again in subsequent draws.

pasting - sampling without replacement

after a datapoint is chosen to be a part of the sample it is cannot be replaced back into the dataset to be picked again in subsequent draws.

out-of-bag evaluation

It can be shown mathematically that 67% of the training instances are used by bagging (with replacement) and rest 33% are not used (for a single classfier but i can be used by other classifiers in the ensemble).

This 33% can be used as testing data.

With enough estimators the whole training data can be then used as testing data also

Random patches and random subspaces

sampling of features

random forest

ensemble of decision trees only

selects the best feature among a random subset of features

boosting 

hypothesis boosting - combine several weak learners into a strong learner

In boosting methods we train predictors sequentially

So that new predictor models can learn something from old predecessors

adaptive boosting - adaboost

pay a bit more attention to the training instances that the predecessor underfit

Gradient boost 

 fit the new predictor to the residual errors made by the previous predictor

setting optimal number of trees - GridSearchCV or RandomizedSearchCV

stacking

we train a model to perform aggregation of predictions of all predictors in an ensemble instead of hard voting or soft voting.

Also use a layer of blenders and another layer to agrregate the blenders

---

### Dimensionality Reduction

curse of dimensionality

high-dimensional datasets are at risk of being very sparse

Approaches of DR

Projection

training instances are not spread out uniformly

all training instances lie within (or close to) a much lower-dimensional subspace of the high-dimensional space

manifold learning

to unroll the Swiss roll to obtain the 2D dataset

PCA

Identifies the hyperplane that lies closest to the data, and then it projects the data onto it

PCA identifies the axis that accounts for the largest amount of variance in the training set

Locally linear embedding (LLE)

---

## Unsupervised Learning Techniques

---

 the vast majority of the available data is unlabeled

we have the input features X, but we do not have the labels y

unsupervised learning tasks → clustering, anomaly detection and Density estimation

---

### k-means

given all the instance labels, locate each cluster’s centroid by computing the mean of the instances in that cluster - both centroids and labels are not given

1. place the centroids randomly (randomly select k instances from the training set).
2. Assign each data point to the nearest centroid creating k clusters (eucledian distance).
3. Recalculate the mean of all the data points for each cluster and assign the new point as the centroid. 
4. Repeat 2,3 and 4 till the centroids stop moving.

centroids, similarity score (affinity), blobs

hard clustering 

directly assign the cluster for a node

soft clustering

score per cluster

Centroid initialization methods

setting the init hyperparameter (if you have an idea of where the centroid should be)

run algo multiple times (with different random initialization for the centroid)

Performance metric 

model’s inertia → sum of the squared distances between the instances and their closest centroids

---

### k-means++

smarter initialization step that tends to select centroids that are distant from one another

1. Randomly select the first centroid, say μ1.
2.  Calculate the distance of all points from μ1, then select the second centroid μ2 with a probability proportional to $D(x_i)^2$
I will explain in detail
say the distance to the few point from the current centroid is as follows
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
    - $P(x_2)$ = 4/30 ≈ 0.133P
    - $P(x_3)$ = 9/30 = 0.3
    - $P(x_4)$ = 16/30 ≈ 0.533
    
    So basically we do not always use the farthest point possible but it is based on this probability
    

---

### Accelerated and mini-batch k-means

Better for huge datasets

accelarated k-means → Elkan’a algo

avoiding many unnecessary distance calculations

exploiting the triangle inequality (i.e., that a straight line is always the shortest distance between two points

does not always accelerate training

mini-batch k-means

moving the centroids just slightly at each iteration (mini-batch)

speeds up the algorithm (typically by a factor of three to four)

Finding the optimal number of clusters

Inertia is not a good performance metric when trying to choose k because it keeps getting lower as we increase k.

Silhouette score

The mean silhouette coefficient over all the instances.

An instance’s silhouette coefficient is equal to  $\frac{b-a}{max(a, b)}$,  where a is the mean distance to the other instances in the same cluster (i.e., the mean intra-cluster distance) and b is the mean nearest-cluster distance (i.e., the mean distance to the instances of the next closest cluster, defined as the one that minimizes b, excluding the instance’s own cluster).

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
4. It’s almost like logistic regression, except it uses a step function instead of the logistic function.

A perceptron is composed of one or more TLUs organized in a single layer.

---

### Backpropagation for perceptron

1. The weights of the perceptron are initialized randomly or with small random values.
2. The input data is fed into the network, and calculations are carried out layer by layer from the input layer to the output layer to produce the output.
a. The input values are multiplied by their respective weights.
b. These products are summed, and the sum is passed through an activation function to produce the output.
Mathematically, for a perceptron with inputs x1,x2,...,xn, weights w1,w2,...,wn, and a bias b, the output y is calculated as: y=f(∑i=1nwixi+b) where f is the activation function (e.g., sigmoid, ReLU).
3. The error (or loss) is calculated by comparing the network’s output with  the actual target value using a loss function. A common loss function for a single output is the mean squared error (MSE): Error=12(ypred−ytrue)2 
4. Backpropagation involves three main steps:
a. Calculate the gradient: The gradient of the error with respect to each weight is calculated using the chain rule of calculus. This involves determining how changes in weights affect the error. For the weight wi, the gradient is calculated as: ∂Error∂wi
b. Update the Weights:  The weights are updated in the direction that reduces the error, which is opposite to the gradient. The update rule for the weights is: wi←wi−η∂Error∂wi.
c. Iterate.

---

### Backpropagation for multi-layer perceptron

**Solved Example Back Propagation Algorithm Multi-Layer Perceptron Network by Dr. Mahesh Huddar**

[https://www.youtube.com/watch?v=tUoUdOdTkRw](https://www.youtube.com/watch?v=tUoUdOdTkRw)

[https://chatgpt.com/share/4bec1811-3fc6-45e7-be81-c9e2f31ebad4](https://chatgpt.com/share/4bec1811-3fc6-45e7-be81-c9e2f31ebad4)

Forward pass

Backward pass

Chain rule

---

### Activation functions

**Heaviside**: step function → 0 or 1

**Tanh**: S shaped → -1 to 1 → 

**Sigmoid**: S shaped → 0 to 1 → logistic function → if you need gradient (because step function has no gradient so no progress in gradient descent)
If you want to guarantee that the output will always fall within a given range of values, then use the Sigmoid function.

**Rectified Linear Unit (ReLU)** : to  get positive output only  →  _/ shaped function
If you want to guarantee that the output will always be positive, then use the ReLU activation function.

**Leaky ReLU:**  Slope z < 0 to ensure that leaky ReLU never dies.

**Randomized Leaky ReLU (RReLU):** The slope towards the negative side can be either 0, +ve or -ve depending on a hyperparameter.

**Parametric Leaky ReLU (PReLU):** The slope towards the -ve side is decided by a hyperparameter.

 **Exponential Linear unit (ELU):** performed better than all ReLU’s with lesser training time and better results only disadvantage being it is slower to compute than ReLU.

**Scaled Exponential Linear Unit (SELU):** about 1.05 times ELU

**Gaussian error linear unit** **(GELU):** other activation function discussed above but is computationally intensive.

**Swish and Mish: other** variants of ReLU. 

---

MLP regression architecture → HOML book

MLP classification architecture → HOML book

TensorFlow playground → understanding MLPs → effect of hyperparameters (number of layers, neurons, activation function and more)

---

Keras MLP models

1. Sequential API
2. Functional API

Keras MLP 

How to setup a model Regression or classification any!

1st we will have out normalization layer

next we prepare our neural network using sequential API which takes the normalization layer all hidden layers and output layer

If we have any concatenation layer we add them as well

We compile out model using optimizer, loss and metrics

We fit the model

We evaluate the model

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

Weights initialization → If weights are initialized with high variance even if i/p has low variance the o/p of a layer can have greater variance. The variance of the outputs of each layer is much greater than the variance of its inputs.

Sigmoid function → It squashes the i/p between 0 and 1. If i/p is very large (+ve or -ve) the o/p of the function can be close to 0 or 1.

Therefore the gradient can be close to 0 leading to vanishing gradients.

So how to solve this problem?

So we have to solve 2 problems 

1. weight initialization.
2. using proper activation function.

solving problem 1

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

However solving these two problems still does not ensure that the vanishing/exploding gradients problem does not reoccur during training. To address this we have batch normalization.

Steps in BN:

1. Compute the Mean and Variance: For a given mini-batch, compute the mean and variance of the activations.
$\mu_{batch} = \frac{1}{m} \sum_{i=1}^{m}x_i$

$\sigma_{batch}^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_{batch})^2$
2. Normalize the Activations: Subtract the mean and divide by the standard deviation to normalize the activations.
$\hat{x_i} = \frac{x_i - \mu_{batch}}{\sqrt{\sigma_{batch}^2 + \epsilon}}$
3. **Scale and Shift**: Introduce two trainable parameters, γ (scale) and β (shift), to allow the model to learn the optimal scaling and shifting of the normalized activations..
4. 
$y_i = \gamma \times \hat{x_i} + \beta$ 

There’s no need for StandardScaler or Normalization; the BN layer will do it for you. also BN acts like a regularization thus eliminating the need of regularization techniques.

**Gradient Clipping**

used in RNN for exploding gradients mostly

clip the gradients during backpropagation so that they never exceed some threshold.

---

**Reusing  pretrained models**

1. Transfer learning

If the task at hand is similar to a task that is already solve by a deep neural network (DNN) we can use some or many layer of the existing DNN to help increase the accuracy of the model. We can freeze the reused layers in the first few epochs so that out model are also able to learn and adjust. However it is usally very difficult to find good good configurations and is generally used only in CNNs.

1. Unsupervised Pretraining

If you did not find any model trained on a similar task use GNN or autoencoders (RBMs)

1. Pretraining on an Auxiliary Task
    
    If not much labelled training data is present first train a neural network on an auxiliary task for which you can easily obtain or generate labeled training data, then reuse the lower layers of that network for your actual task.
    

---

### Faster Optimizers

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
The below are variantions of adam only…

1. AdaMax
2. Nadam
3. AdamW

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

1. Performance scheduling

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

1. 1cycle scheduling

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

Early stopping and batch normalization are already acting as regularizers

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
    3. Each forward pass results in a different set of neurons being dropped out, creating an ensemble of N different predictions for each input. After which the mean of the different results and the variance (uncertainty) can we checked which can be used for confidence assesment.
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

### Classification and Localization

This can be expressed as a regression task, the aim is to predict a bounding box around the image.

4 numbers are to be predicted → horizontal and vertical coordinates of the object’s center, as well as its height and width.

So to the same CNN model that were prepared above we just need to add a second dense output layer with four units typically on top of the global average pooling layer.

Intersection over Union (IoU) can be used instead of MSE as metric for evaluation

### Object detection

A sliding CNN is used to detect multiple objects

But this detects the same object multiple times so the unnecessary boxes can be eliminated using non-max suppression.

1. Fully Convolutional Network (FCN).
2. You Only Look Once (YOLO).

### Object tracking

1. DeepSORT

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

### Handling Long Sequences

**The Unstable Gradients Problem**

**Tackling the Short-Term Memory Problem**

**LSTM cells**

**GRU cells**

---

## Natural Language Processing with RNNs and Attention

---

## Autoencoders, GANs, and Diffusion Models

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
    
    shortest path distance between two nodes
    
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
    1. Katz index.
        
        Count the number of paths of all lengths between a given pair of nodes by using powers of graph adjacency matrix.
        
        
    

---

### **Graph Neural Networks**

“message passing neural network” framework proposed → Neural Message Passing for Quantum Chemistry by Gilmer er al using  Graph Nets architecture schematics introduced by Battaglia et al.

GNNs adopt a “graph-in, graph-out” architecture meaning that these model types accept a graph as input, with information loaded into its nodes, edges and global-context, and progressively transform these embeddings, without changing the connectivity of the input graph. 

**The simplest GNN**

We can use a separate MLP on each component of the graph → making it a GNN layer

For each node vector we apply MLP and get a learned node-vector, doing the same for each edge and global context-vector. [only ouput will have different embeddings].

**Pooling**

Information can be stored in edges but not in nodes.

Collecting information from nodes and giving it to edges → Pooling → Aggregate information from adjacent edges.

vice-versa for collection info from edges to nodes.

**Message Passing**

3 steps of message passing

1. For each node in the graph, gather all the neighboring node embeddings (or messages).
2. Aggregate all messages via an aggregate function (like sum).
3. All pooled messages are passed through an update function, usually a learned neural network.

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