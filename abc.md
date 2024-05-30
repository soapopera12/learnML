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

What is a limit of a function?

It describes the behavior of a function as its input approaches a certain value

[https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3](https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3)

What are the rules of differentiation?

[https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3](https://chatgpt.com/share/2e41d2d1-6635-4a51-951b-7f89d755dfb3)

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

### DBSCAN

density-based spatial clustering of applications with noise (DBSCAN)

ε (epsilon), min_samples,  ε-neighborhood, core instance

1. For each instance, the algorithm counts how many instances are located within a small distance ε (epsilon) from it. This region is called the instance’s ε-neighborhood.
2. If an instance has at least min_samples instances in its ε-neighborhood (including itself), then it is considered a core instance.
3. All instances in the neighborhood of a core instance belong to the same cluster.
4. Any instance that is not a core instance and does not have one in its neighborhood is considered an anomaly.

### Gaussian Mixtures

A probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown.

---

## Artificial Neural Network

### The perceptron

Threshold logic unit (TLU)

1. The inputs and output are numbers (instead of binary on/off values), and each input connection is associated with a weight.
2. The TLU first computes a linear function of its inputs.
3. Then it applies a step function to the result.
4. It’s almost like logistic regression, except it uses a step function instead of the logistic function.

A perceptron is composed of one or more TLUs organized in a single layer.

### Multi-Layer perceptron and backpropagation

1. It handles one mini-batch at a time (for example, containing 32 instances each), and it goes through the full training set multiple times. Each pass is called an epoch.
2. Each mini-batch enters the network through the input layer. The algorithm then computes the output of all the neurons in the first hidden layer, for every instance in the mini-batch. The result is passed on to the next layer, its output is computed and passed to the next layer, and so on until we get the output of the last layer, the output layer. This is the forward pass: it is exactly like making predictions, except all intermediate results are preserved since they are needed for the backward pass. 
3. Next, the algorithm measures the network’s output error (i.e., it uses a loss function that compares the desired output and the actual output of the network, and returns some measure of the error).
4. Then it computes how much each output bias and each connection to the output layer contributed to the error. This is done analytically by applying the chain rule (perhaps the most fundamental rule in calculus), which makes this step fast and precise.
5. The algorithm then measures how much of these error contributions came from each connection in the layer below, again using the chain rule, working backward until it reaches the input layer. As explained earlier, this reverse pass efficiently measures the error gradient across all the connection weights and biases in the network by propagating the error gradient backward through the network (hence the name of the algorithm).
6. Finally, the algorithm performs a gradient descent step to tweak all the connection weights in the network, using the error gradients it just computed.

In short, backpropagation makes predictions for a mini-batch (forward pass), measures the error, then goes through each layer in reverse to measure the error contribution from each parameter (reverse pass), and finally tweaks the connection weights and biases to reduce the error (gradient descent step).

hidden layers  :                      Depends on the problem, but typically 1 to 5

neurons per hidden layer  :  Depends on the problem, but typically 10 to 100

output neurons   :                1 per prediction dimension

Hidden activation   :             ReLU

Output activation  :              None, or ReLU/softplus (if positive outputs) or sigmoid/tanh (if bounded outputs)

Loss function  :                     MSE, or Huber if outliers

RNN

CNN

Autoencoders

GANs

Diffusion models

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