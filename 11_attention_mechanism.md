# Attention mechanism

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

## KV Cache

For models with long context windows (32k, 128k, or even 1M+ tokens), inference is performed via auto-regressive decoding, where the model generates one token at a time. At each decoding step, the new token must attend to all previously generated tokens, requiring computation to their corresponding key and value representations. Once naive way would be to to generate token key and value for each 1 to n-1 th token which computation of self attention of current nth node. This could be very slow and increases compute- and memory-intensive. Another better way could be using K-V cache solution where we pre-compute and store all the Keys and Value vectors once per token per layer and store them in GPU memory forever. And when computing self attention for new token we only compute its query and do matrix multiplication with with all previous keys and value.

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
3. Position 1 and position 1000 might look similar (check values of sin 1 radian and sin 1000 radian). Else check for sin(180$^\circ$) and sin(540$^\circ$) (in degrees)
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

Instead of calculating a position vector separately for each embedding we simply rotate the word embedding vector by some angle by applying a matrix transformation that embeds the information of the position along with the word embedding by simply rotating the word embedding vector. RoPE treats the d-dimensional embedding as a set of d/2 pairs of coordinates. For each pair, it applies a rotation based on the position m of the token.

If we have a 2D vector [$x_1$,$x_2$], the rotated version at position m is:

$$
\begin{bmatrix}
    x_1' \\
    x_2'
\end{bmatrix} = 
\begin{bmatrix}
    \text{cos (m} \theta ) & -\text{sin (m} \theta)\\
    \text{sin (m} \theta ) & \text{cos (m} (\theta )
\end{bmatrix} \begin{bmatrix}
    x_1 \\
    x_2
\end{bmatrix}
$$

When the model performs the Attention mechanism, it calculates the dot product between a Query (Q) and a Key (K).

$$ \text{Attention Score} = \text{Rotated} \ Q_m \cdot \text{Rotated} \ K_m$$

Because of the way rotation works in 2D space, the dot product of two rotated vectors is mathematically equivalent to the dot product of the original vectors, but adjusted by the cosine of the angle between them.

$$Q_m \cdot K_n \propto cos(m \theta - n \theta) = cos((m-n)\theta)$$

This calculation does not actually happen it is just the theoretical concept which is encoded into the idea of RoPE. The simple matrix transformation of RoPE is that when you calculate the dot product (attention score) between two tokens at positions m and n, the result only depends on the relative distance m−n. As tokens get further apart, their vectors are rotated further away from each other, naturally signaling to the model that they are less related.

