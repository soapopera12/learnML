# Transformers

## Working of a transformer

Transformers consists of two parts an encoder and a decoder. The encoding part is a group of multiple encoders and the decoding part is a set of multiple decoders.

---

### Encoder

An encoder consists of a self-attention layer followed by a feed forward neural network layer. 

<p align="center">
    <img src="pics/Transformer_encoder.png" alt="Transformer_encoder" width="500"><br>
    <em>Encoder</em>
</p>

---

### Decoder

The decoder on the other hand has an additional layer that focuses helps it focus on the relavant part of the input sentences.

<p align="center">
    <img src="pics/Transformer_decoder.png" alt="Transformer_decoder" width="500"><br>
    <em>Decoder</em>
</p>

---

### Transformer input

The input to the transformer encoder is a sentence which is a group of words. These words are converted into a embeddings using an embedding algorithm (word2vec, glove, etc). Each word vector embedding is of same size (a hyperparameter eg. 512). Each of these embeddings pass through an encoder. Each embeddings first pass through the self attention block followed by the FFN network

<p align="center">
    <img src="pics/encoder_with_tensors.png" alt="tensor_encoder" width="500"><br>
    <em>Encoder with tensors</em>
</p>

---

### Self attention

<p align="center">
    <img src="pics/transformer_self_attention_vectors.png" alt="transformer_self_attention_vectors" width="500"><br>
    <em>Self attention</em>
</p>


1. From the encoder's input create three vectors Query, Key and Value ($E_Q$, $E_K$ and $E_V$). This is formed by multiplying the input vector with three matrices that are learned during training ($W^Q$, $W^K$ and $W^V$). 
2. Next we calculate a score. This score needs to be calculated for each word in the input to the (it is a score of attention that each word should give to each other word). It is calculated by taking a dot product of the query vector with the key vector of the respective word that we are scoring.
eg:- The hill station is chilly. Here suppose we are at word station. The we to find score for chilly and station (current word) we need to multiply the query of 'station' to key of 'chilly'.
4. We then divide the score by 8 (basically $\sqrt{d_k}$ here $d_k$ is 512 so 8) To ensure stable gradients. 
4. The result is then passed through a softmax operation (all scores are normalized so that they add to 1). This softmax score determines how much each word will be expressed at this current position. 
5. Now we multiply **each** value vector by the softmax score (this keeps intact the value of words that needs to be attended to at each point). 
6. The sixth step is to sum all the weighted value vectors. This produces the output of the self-attention layer at this position. 

<p align="center">
    <img src="pics/self-attention-output.png" alt="self-attention-output" width="500"><br>
    <em>Self attention calculation</em>
</p>

---

### Matrix calculation of self-attention (every input at once)

The first step is the calculation of Query, Key and Value matrices. We do this by packaging out embeddings into a matrix X, and multiplying it with weight matrices we've trained ($W_Q, W_K, W_V$)

<p align="center">
    <img src="pics/self-attention-matrix-calculation.png" alt="self_attention_matrix_calculation" width="500"><br>
    <em>Self attention matrix calculation</em>
</p>

Finally, since we’re dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer (each query row automatically mulitplies with each Key resulting in a matrix of scores followed by mulitplication with each value to give final z embeddings which can be added).

<p align="center">
    <img src="pics/self-attention-matrix-calculation-2.png" alt="Self attention matrix calculation formula" width="500"><br>
    <em>unwrapping the sin for wave representation</em>
</p>

In the case of multiple heads we get multiple V matrices at the end. These matrices are concatenated are multiplied by a new matrix $W_O$ which has resulting dimension of original V matrix.

---

### Positional encoding

This bit attends to giving the embedding/word the positional information of itself and its peer words in the sentence. This is calculated using cosine and sine formulas and added to input embeddings.

---

### Residual connections

The input to an encoder is used as a residual connection at the end of self attention block at the start of add and normalize block.

<p align="center">
    <img src="pics/transformer_resideual_layer_norm_2.png" alt="Transformer_residual_layer_normalization" width="500"><br>
    <em>Transformer residual layer normalization</em>
</p>

---

### decoder

In decoder the self attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking the future positions before the softmax step in self-attention calculation. This decoder works similar to encoder only difference would be that it creates its query matrix from the layer below it but takes the keys and values from the output of the encoder stack (this is called encoder-decoder attention).

---

## T5

The T5 model (short for Text-to-Text Transfer Transformer) showed that scaling up transformers (hundreds of millions to billions of parameters) and pretraining them on massive data (C4 dataset) gave strong generalization. Instead of BERT’s masked language modeling (masking individual tokens), T5 used span corruption:

1. Random spans of text are replaced with a unique sentinel token (like <extra\_id\_0>).
2. The model learns to generate the missing spans.
3. example:
    1. Input → "The <extra\_id\_0> is on the <extra\_id\_1>."
    2. Output → "<extra\_id\_0> book <extra\_id\_1> table"


It can be used for Summarization, translation, classification, QA, reasoning etc. Chatmodels like chatGPT require long form generation and dialogue memory, however T5 models are less chatty and harder to scale at 100B+ params.

---