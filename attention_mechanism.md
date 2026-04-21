# Attention mechanism

## Standard Attention and Multi-Headed Attention (MHA)

Attention mechanism is the core principle of transformer where a token finds the amount of attention that it should attend to the words appearing before and after itself. The auto-regressive decoding network or decoder of a transformer generates one token at a time, for the newly generated token we use the self attention block to generate the Q, K and V from $W_Q$, $W_K$ and $W_V$ matrices followed by the computation of attention of all previous tokens Keys and the current tokens Query using the formula:

$$
Q = XW_Q
$$
$$
K = XW_K
$$
$$
V = XW_V
$$

$$
Attn(Q, K, V) = softmax(\frac{QK^T}{\sqrt(d_k)})V
$$

Remember here that the current tokens Query is generated and that is multiplied by the previous tokens K followed by softmax and then multiplied by previous node's V. This is done once for all the previous token to get pairwise attention scores. This can happen in a single headed model or MHA model. The vanilla attention mechanism expects that for each word we compute the query key and value every-time we encounter it at every level which can be time consuming and is the naive way.

So the storage consideration for models Keys and Values can be considered at
$$ 
= \text{batch\_size} \times \text{sequence length} \times \text{no of layers} \times \text{num of heads} \times \text{head dimension} \times \text{2 (for K and V)} 
$$

however this can cause problems at large scaler

1. KV cache memory explosion
2. Numerical instability
3. poor extrapolation to long context
4. inference latency bottlenecks
5. routing instability in MoE networks

---

## Cross-Attention

**core idea**
Cross-attention is a mechanism where one sequence attends to another sequence.
Instead of deriving everything from the same input, the model uses information from a different source.

In attention, we have:

* Query (Q)
* Key (K)
* Value (V)

The key difference is where these come from.

**In cross-attention**

* Query (Q) comes from one sequence (e.g., decoder)
* Key (K) and Value (V) come from another sequence (e.g., encoder)

The computation remains the same:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

But now:

* Q ≠ K, V source

**Intuition and where it is used**

1. Encoder–Decoder Transformers
   In models like Attention Is All You Need

* Encoder processes input sentence
* Decoder generates output
* Decoder uses cross-attention to “look at” encoder outputs

    Example:
    While generating a translated word, the decoder attends to relevant words in the input sentence.

2. Multimodal Models
   In models like CLIP

* Text queries attend to image features
* Enables understanding across modalities

    Example:
    A caption attends to different parts of an image.

3. Retrieval-Augmented Generation (RAG)

* Query attends to retrieved documents
* Model conditions its output on external knowledge

    Example:
    A question attends to relevant passages before generating an answer.

Cross-attention is what allows transformers to connect different sources of information and align them during generation.

---

## Causal Attention (Masked Attention)

**core idea**
Causal attention is a mechanism where each token can only attend to **previous tokens and itself**, not future tokens. This enforces a left-to-right (autoregressive) structure during generation. Used in GPT (decoder) models.

In attention, we still have:

* Query (Q)
* Key (K)
* Value (V)

But we restrict *which tokens are visible*.

**In causal attention**

* Q, K, V all come from the same sequence (like self-attention)
* A **mask** is applied so position *i* cannot see positions *> i*

The computation becomes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
$$

Where:

* ( M ) is the mask
* Future positions get ( $-\infty$ ) (so their softmax becomes 0)

**Intuition and where it is used**

1. Autoregressive Language Models
   In models like GPT-4

* Tokens are generated one by one
* Each token can only depend on what came before

    Example:
    When predicting the next word in
    “the cat sat on …”
    the model cannot look at the future word.

2. Text Generation (Next Token Prediction)

* Training objective: predict next token
* If future tokens were visible, the task becomes trivial

    Causal masking ensures:

* Proper learning of sequence dependencies
* No information leakage from the future

3. KV Cache and Efficient Inference

* During generation, past keys and values are cached
* New tokens only attend to previous tokens

This works naturally because:

* future tokens are never needed
* attention is strictly one-directional

Causal attention is what makes transformers behave like **sequence generators**, ensuring they produce text step-by-step without cheating by looking ahead.

---

## Linear Attention

**core idea**
Linear attention is an **approximate version of standard attention** that is redesigned to reduce computation from quadratic to linear time.
It removes softmax (or approximates it using kernel feature maps) and **changes how interactions are computed** so that we never explicitly form the full pairwise attention matrix.

It introduces a feature map $ \phi(\cdot) $, which is:

* A **kernel transformation**
* Usually **fixed (non-learned)**
* Often a **simple element-wise function**

Example:

$$
\phi(x) = \text{ELU}(x) + 1
$$

Key idea:

$$
\text{softmax}(QK^T) \approx \phi(Q)\phi(K)^T
$$

This makes attention **factorizable** and enables reordering of computation.

**Standard attention computation:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

Step 1:

$$
Q (n \times d) \cdot K^T (d \times n) = (n \times n)
$$

Cost:

$$
O(n^2 d)
$$

Step 2:

$$
(n \times n) \cdot (n \times d) = (n \times d)
$$

Cost:

$$
O(n^2 d)
$$


So it becomes **matrix × matrix**, giving quadratic complexity.

**Linear attention computation:**

Instead of forming:

$$
QK^T \quad (n \times n)
$$

we compute:

$$
\phi(K)^T V
$$

Shapes:

$$
(d \times n) \cdot (n \times d) = (d \times d)
$$

Cost:

$$
O(n d^2)
$$

Then:

$$
\phi(Q) (n \times d) \cdot (d \times d) = (n \times d)
$$

Cost:

$$
O(n d^2)
$$

**why it works for efficiency**

* Uses associativity to reorder computation
* Avoids constructing the $ n \times n $ attention matrix
* Reduces time from $ O(n^2 d) $ to $ O(n d^2) $
* Reduces memory from $ O(n^2) $ to $ O(n) $
* Works when attention can be approximated via kernel feature maps

**where it is used**

1. Performer
   Uses random feature maps to approximate softmax attention efficiently

2. Linformer
   Projects keys and values to lower dimensions to reduce attention cost

---

## Sliding Window Attention (Local Attention)

**core idea**
Sliding window attention is a **restricted version of standard attention** where each token attends only to a **fixed local neighborhood** instead of all tokens.
It keeps softmax attention unchanged but **limits interactions to nearby tokens**, avoiding the full ( n \times n ) attention matrix.

Standard attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

Cost:

$$
O(n^2)
$$

because every token attends to all ( n ) tokens.

Sliding window attention:

Each token attends only to a window of size ( w ):

$$
\text{Attention}(Q_i, K_{i-w:i+w}, V_{i-w:i+w})
$$

So instead of full pairwise attention:

* Each token interacts with only ( w ) neighbors

Cost becomes:

$$
O(n \cdot w)
$$

**where it is used**

1. Longformer
   Uses sliding window + some global tokens for long documents

2. BigBird
   Combines local window + random + global attention

---

## Global Attention

**core idea**
Global attention is a pattern where a **subset of tokens is given full visibility over the entire sequence**, meaning these tokens can attend to all tokens and all tokens can attend to them, while the rest of the tokens may still use restricted (e.g., local) attention. This allows the model to maintain **global context efficiently** without incurring full $ O(n^2) $ cost for every token, since only a few selected tokens (like special or important positions) perform full attention, acting as information hubs that aggregate and distribute context across the sequence.

---

## KV Cache

For models with long context windows (32k, 128k, or even 1M+ tokens), inference is performed via auto-regressive decoding, where the model generates one token at a time. At each decoding step, the new token must attend to all previously generated tokens, requiring computation to their corresponding key and value representations. Once naive way would be to to generate token key and value for each 1 to n-1 th token which computation of self attention of current nth node. This could be very slow and increases compute- and memory-intensive. Another better way could be using K-V cache solution where we pre-compute and store all the Keys and Value vectors once per token per layer and store them in GPU memory forever. And when computing self attention for new token we only compute its query and do matrix multiplication with with all previous keys and value.

---

## Paged Attention (KV Cache Memory Management)

**core idea**
Paged attention is a technique to **efficiently manage KV cache memory** during inference by dividing it into **fixed-size blocks (pages)** instead of storing it as one large contiguous chunk per sequence.
It avoids memory fragmentation and allows dynamic allocation, so multiple sequences can share GPU memory efficiently.

**what problem it solves**

In standard KV cache:

* Each sequence stores its Keys and Values as it grows

$$
\text{KV cache size} \sim O(n \cdot d)
$$

Problem:

* Sequences have **different lengths**
* Memory must be allocated as one continuous block (**very difficult to manage**)
* Leads to:

  * Wasted space
  * Fragmentation
  * Difficult resizing

**paged attention idea**

Instead of:

Sequence A → [continuous memory block]
Sequence B → [continuous memory block]

We break memory into **fixed-size pages**:

Page size = fixed (e.g., 16 tokens per page)

Sequence A → Page1, Page5, Page9
Sequence B → Page2, Page3

Pages are **not required to be contiguous**

**how attention works with pages**

When computing attention for a token:

$$
\text{Attention}(Q_t, K_{1:t}, V_{1:t})
$$

Instead of reading from one continuous block:

* The system **gathers relevant pages**
* Concatenates them logically (not physically)
* Computes attention normally

**why it works for efficiency**

* Eliminates memory fragmentation
* Allows dynamic growth of sequences
* Reuses freed pages when sequences finish
* Enables continuous batching (many users at once)
* Improves GPU memory utilization significantly

**key difference vs normal KV cache**

Normal KV cache:

* One large block per sequence
* Hard to resize
* Wastes memory

Paged attention:

* Small fixed blocks (pages)
* Flexible allocation
* Memory reuse across sequences

**where it is used**

1. vLLM
   Introduced paged attention for efficient large-scale inference

---

## Grouped Query Attention mechanism (GQA)

This attention mechanism tries to use as many query heads as we have and fewer (shared) Key and Value heads. So basically the number of queries will be equal to the number of heads but we pair the Key and Value pair for each group. eg:- For a model with 32 query heads and 8 groups of keys and 8 values heads. Each Key and Value is shared across 4 query heads. this saves KV memory, reduces bandwidth and improves stability. So now 4 query heads will share the same Key and Value this will reduce the count of keys and queries over multiple heads reducing storage size.

So the storage consideration for models Keys and Values can be considered at
$$ 
= \text{batch\_size} \times \text{sequence length} \times \text{no of layers} \times \text{num of KV heads} \times \text{head dimension} \times \text{2 (for K and V)} 
$$

---

## Multi-headed latent attention (MLA)

This attention mechanism tries to use an encoder-decoder network that learns how to compress K and V vectors for each attention head. Once this model learns how to encode the K and V vectors into a compressed latent dimension and stores this into KV cache storage. Whenever required the K and V are restored into its original dimension to calculate MHA. This must be used with decoupled ROPE as latent dimension loses positional information. 

So the storage consideration for models Keys and Values can be considered at
$$
= \text{batch\_size} \times \text{sequence length} \times \text{no of layers} \times \text{latent dim + ROPE dim} 
$$

---

## QK-normalization

In the technique attention score are controlled and kept in range so that the router and attention don't destabilize each other (In MoE transformer architecture)

$$
\hat{Q} = \frac{Q}{||Q||}
$$

$$
\hat{K} = \frac{K}{||K||}
$$

$$
 = \hat{Q} . \hat{K}
$$

---

## Absolute Positional Encoding

Input to Model = $E_i$ + $P_i$

$E_i$ : The semantic meaning (e.g., the concept of "Cat").
$P_i$ : A fixed vector representing "Position 2".

Learned Embeddings (BERT/GPT-2): The model treats positions like a vocabulary. It "learns" a specific vector for "Position 1," another for "Position 2," etc., during training.

---

## Sinusoidal Positional Embeddings

Here we want to convert a position number into an embedding

pos could be = [1, 2, 3, 4, ....]

For this sin/cosine can be used because they are continuous and encodes distance.

sin(5) and sin(6) are close (in radians)

sin(5) and sin(100) are not

However one sine wave is not enough

If we used only:

1. PE(pos) = sin(pos)
2. Then, Positions repeat every 2π
3. Position 1 and position 1000 might look similar (check values of sin 1 radian and sin 1000 radian). Else check for sin(180 $^\circ$) and sin(540 $^\circ$) (in degrees)
4. So we use many sine waves at different frequencies.

The position embedding for a token is represented by a vector of size d same as the size of the embedding vector. For position embedding we use sine-cosine pair so

Dimension 0 $\rightarrow$ sine

Dimension 1 $\rightarrow$ cosine

Dimension 2 $\rightarrow$ sine

Dimension 3 $\rightarrow$ cosine

This part can be better understood with an example

Suppose the embedding vectors are of size d = 8, So position embedding vector needs to be of the same size so d = 8.

Assume the sentence is 

$$
\text{I am drake}
$$ 

Now we know that d = 8 and position of each word (0 to 2). So for each word we need to create a vector of length d (positional embedding vector)

for word 'drake' it goes like this

$$
d = 8
$$
$$
pos = 2 \ \ \text{for word = 'drake'}
$$

then the positional encoding vector is


| Dimension | Value |
|---|---|
|0 | $sin(pos/10000^0)$ |
|1 | $cos(pos/10000^0)$ |
|2 | $sin(pos/10000^{(2/8)})$ |
|3 | $cos(pos/10000^{(2/8)})$ |
|4 | $sin(pos/10000^{(4/8)})$ |
|5 | $cos(pos/10000^{(4/8)})$ |
|6 | $sin(pos/10000^{(6/8)})$ |
|7 | $cos(pos/10000^{(6/8)})$ |

**Table: Positional Encoding Vector (d_model = 8)**

All this for a single word 'drake'. This needs to be done for all the words. So what does it actually do. i is just the index of the (sin, cosine) pair that we are using.


| i | Dimension used |
|---|---|
0 | (0, 1) 
1 | (2, 3) 
2 | (4, 5) 
3 | (6, 7) 

**Table: index of dimensions for pos embedding**

Each pair uses the same frequency. And encodes position at different scale (fast -> slow). So i is the frequency. The dimensions are grouped as pairs

$$
(0,1),(2,3),(4,5),…
$$

The main formula can we written in a clearer way

$$
PE(pos, 2i) = sin(\frac{pos}{10000^{2i/d}}) \ \ \text{for even indices}
$$

$$
PE(pos, 2i+1) = cos(\frac{pos}{10000^{2i/d}}) \ \ \text{for odd indices}
$$

---

## Rotary Positional Embedding (RoPE)


Instead of calculating a position vector separately for each embedding, we rotate the word embedding vector by some angle using a matrix transformation. This embeds positional information directly into the word embedding.

In practice, RoPE is applied to the Query (Q) and Key (K) vectors — not directly to the raw token embeddings (very important).

RoPE treats a d-dimensional embedding as **d/2 pairs of coordinates**. For each pair, it applies a rotation based on the position *m* of the token.

If we have a 2D vector [$x_1$, $x_2$], the rotated version at position *m* is:

$$
\begin{bmatrix}
x_1' \
x_2'
\end{bmatrix}

=

\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\begin{bmatrix}
x_1 \
x_2
\end{bmatrix}
$$

**1. What RoPE actually does**

1. For each pair of dimensions:
   $$
   (x_1, x_2) \rightarrow (x_1', x_2') = R(m\theta) \cdot (x_1, x_2)
   $$

2. For token at position *m*, rotation angle:
   $$
   \text{angle} = m \cdot \theta
   $$

3. Different dimension pairs use different values of θ (frequencies)

**2. The important shift in thinking**

1. Classic positional embeddings:

   * Add position into the vector

2. RoPE:

   * Does not add anything
   * Changes how vectors interact with each other

**3. Where position actually shows up**

1. Attention score:
   $$
   \text{score} = Q_m \cdot K_n
   $$

2. After applying RoPE:
   $$
   Q_m' = R(m\theta)Q
   $$
   $$
   K_n' = R(n\theta)K
   $$

3. Dot product:
   $$
   Q_m' \cdot K_n' = (R(m\theta)Q) \cdot (R(n\theta)K)
   $$

**4. The key property of rotation**

1. Rotation matrices satisfy:
   $$
   R(a)^T R(b) = R(b - a)
   $$

   Here R is matrix and a is a scalar value by which the angle gets rotated. Similarly for b. and vectors like Q and K are multiplied by these Q` = ($R(a)$)Q. Both R(a) and R(b) are rotations in the same space, but by different angles
   
   where a = m $\theta$, and m is the position of the current word.

   $$
   \begin{bmatrix}
   \cos(a) & -\sin(a) \\
   \sin(a) & \cos(a)
   \end{bmatrix}
   $$

2. So:
   $$
   Q_m' \cdot K_n' = Q \cdot R((n - m)\theta)K
   $$

**5. What this means**

1. Attention depends on:
   $$
   (n - m)
   $$

2. NOT on absolute positions

3. RoPE naturally encodes **relative position**

**6. So where is position encoded?**

1. Position is encoded in:

   * Relative angle differences between tokens

2. Not stored inside the vector explicitly

3. Instead:

   * Rotation changes how vectors align with each other

**7. Intuition**

1. Think of embeddings as points/vectors in space

2. Position effect:

   * Position 0 → no rotation
   * Position 1 → small rotation
   * Position 10 → larger rotation

3. Interaction:

   * Nearby tokens → small angle difference → high similarity
   * Far tokens → large angle difference → low similarity

4. Key idea:

   * Distance between tokens = angle difference

**8. Example (embedding dimension = 8)**

1. Suppose embedding:
   $$
   x = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]
   $$

2. Split into 4 pairs:
   $$
   (x_1, x_2), (x_3, x_4), (x_5, x_6), (x_7, x_8)
   $$

3. Each pair uses a different θ:

   * θ₁, θ₂, θ₃, θ₄

**Example for a word at position m = 3**

Here pos = 3

1. First pair:
   $$
   (x_1, x_2) \rightarrow R(3\theta_1)(x_1, x_2)
   $$

2. Second pair:
   $$
   (x_3, x_4) \rightarrow R(3\theta_2)(x_3, x_4)
   $$

3. Third pair:
   $$
   (x_5, x_6) \rightarrow R(3\theta_3)(x_5, x_6)
   $$

4. Fourth pair:
   $$
   (x_7, x_8) \rightarrow R(3\theta_4)(x_7, x_8)
   $$

**Final rotated embedding**

$$
x' = [x_1', x_2', x_3', x_4', x_5', x_6', x_7', x_8']
$$

**Matrix identity formula proof**


**1. Define the rotation matrix**

$$
R(a) =
\begin{bmatrix}
\cos a & -\sin a \
\sin a & \cos a
\end{bmatrix}
$$

$$
R(b) =
\begin{bmatrix}
\cos b & -\sin b \
\sin b & \cos b
\end{bmatrix}
$$

**2. Take transpose of (R(a))**

Transpose means swap rows and columns:

$$
R(a)^T =
\begin{bmatrix}
\cos a & \sin a \
-\sin a & \cos a
\end{bmatrix}
$$

**3. Multiply (R(a)^T R(b))**

$$
R(a)^T R(b) =
\begin{bmatrix}
\cos a & \sin a \
-\sin a & \cos a
\end{bmatrix}
\begin{bmatrix}
\cos b & -\sin b \
\sin b & \cos b
\end{bmatrix}
$$

**4. Do the matrix multiplication**

Top-left element:

$$
\cos a \cos b + \sin a \sin b = \cos(a - b)
$$

Top-right element:

$$
-\cos a \sin b + \sin a \cos b = \sin(a - b)
$$

Bottom-left element:

$$
-\sin a \cos b + \cos a \sin b = -\sin(a - b)
$$

Bottom-right element:

$$
\sin a \sin b + \cos a \cos b = \cos(a - b)
$$

**5. Final matrix**

$$
R(a)^T R(b) =
\begin{bmatrix}
\cos(a - b) & \sin(a - b) \
-\sin(a - b) & \cos(a - b)
\end{bmatrix}
$$

**6. Recognize the result**

This is exactly:

$$
R(b - a)
$$

because $cos(a-b)=\cos(b-a)$ and $sin(a-b)=-\sin(b-a)$


**7. Final result**

$$
R(a)^T R(b) = R(b - a)
$$

**8. Key insight**

1. Transpose = inverse:
   $$
   R(a)^T = R(-a)
   $$

2. Rotations add:
   $$
   R(-a)R(b) = R(b-a)
   $$

---
