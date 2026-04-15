# Large language models

**Types of LLMS:**

1. Autoencoder-Based Model
2. Sequence-to-Sequence Model
3. Transformer-Based Models
4. Recursive Neural Network Models
5. Hierarchical Models

How do LLMs work

1. **Word Embedding**: This involves representing words as vectors in a high-dimensional space where similar words are grouped together. Creating word embeddings involves training a neural network on a large corpus of text data, such as news articles or books. During training, the network learns to predict the likelihood of a word appearing in a given context based on the words that come before and after it in a sentence. The vectors that are learned through this process capture the semantic relationships between different words in the corpus. A similar approach is followed with the words "King," "Queen," "Man," and "Woman."
2. **Positional Encoding**: Positional encoding is all about helping the model figure out where words are in a sequence.
3. **Transformers**: The transformer layer works by processing the entire input sequence in parallel rather than sequentially. The self-attention mechanism allows the model to assign a weight to each word in the sequence, depending on how valuable it is for the prediction. 
4. **Text  Generation**: Text generation relies on a technique called autoregression, where the model generates each word or token of the output sequence one at a time based on the previous words it has generated.
5. **Empowering AI Performance With Human-Guided Reinforcement Learning**: RLHF implies a form of continuous feedback provided by a human entity to the machine learning model. The feedback can be either explicit or implicit. In the case of LLMs, if the model does return a wrong answer, the human user can correct the model, improving the model’s overall performance.

---

## LLM Components

###  Input Context

In an LLM, the input context is everything the model can “see” when generating the next token. For example, if you ask: [who, is, the, pm, of, india, ?] .The model might reply: [The, PM, of, India, is, Narendra, Modi, .]

Each generated token (like “The”) is both part of the output to you and is also fed back as input so the model can predict the next one. In a chat, the conversation history is also kept in the input context. So your next question becomes: [who, is, the, pm, of, india, ?, The, PM, of, India, is, Narendra, Modi, ., explain, theory, of, relativity, ?]

So the old conversation is preserved and fed back into the input window along with your new question. That’s why we call it a context window — everything the model “remembers” about the conversation lives inside that rolling input. If the conversation gets too long (more than 128k tokens in your example), the oldest parts get truncated or summarized before continuing.

---

###  Tokenizers

LLM tokenizers are trained through a process that involves analyzing large amounts of text data to determine the most efficient way to split words, subwords, or characters into tokens. The goal is to optimize for both vocabulary size and model efficiency. 

**Types of tokenizers**:

1. Byte Pair Encoding (BPE) (Used in GPT-2, GPT-3)
2. WordPiece (Used in BERTn RoBERTa)
3. Unigram Language Model (Used in XLNet, T5, ALBERT)
4. SentencePiece (Used in T5, mT5, LLaMA-2)
5. Character-based or Byte-level Tokenizers (Used in GPT-4’s tiktoken)

**Training tokenizers**:

1. **BPE**:
    1. A large corpus of text is gathered, this corpus is preprocessed by normalizing text (e.g., lowercasing, removing special characters) depending on the tokenizer type.
    1. Start with a base vocabulary of individual characters or bytes.
    1. Compute the frequency of adjacent character pairs in the corpus.
    1. Merge the most frequent pair into a new token.
    1. Repeat until reaching the desired vocabulary size.
2. **WordPiece**: Similar to BPE but uses likelihood maximization to decide which tokens to merge.
3. **SentencePiece**: Can implement unigram or BPE.
4. **Unigram LM**: Starts with a large vocabulary and gradually removes tokens that contribute the least to model efficiency.
5. **Byte-level Tokenizers**: Works at byte level instead of characters → can encode any text. Initially text is converted to bytes. Then we count byte pair frequencies and merge the most frequent pairs. The resulting vocab contains single bytes for rare characters and for freq occurring byte sequences.

The process continues until a predefined vocabulary size is reached (e.g., 50k tokens for many modern LLMs). Special tokens (e.g., [PAD], [CLS], [SEP]) are added for model-specific needs. The final tokenizer is stored as a mapping of tokens to unique integer IDs. These tokens are then saved in a json file as a lookup table

---

###  Embedding matrix

**Simple embedding matrix**



**Token** | **Token ID** | **Embedding Vector (size = 4)**
|---|---|---|
"the" | 1 | [0.12, -0.56, 0.89, 0.34] 
"cat" | 2 | [0.76, 0.44, -0.98, 0.67] 

Note: In reality, embedding dimensions (vector size) are much larger—usually 768 for BERT, 4096 for GPT-4.

1. **How is the Embedding Matrix Formed?**

    1. **Step 1**: Tokenizer Converts Text to Token IDs
    
        e.g. ["the", "cat", "is", "playing"] → [1, 2, 5, 4]
    
    2. **Step 2**: Token IDs are Mapped to Embedding Vectors
    
        Each token ID is mapped to a row in the embedding matrix

        1 → [0.12, -0.56, 0.89, 0.34]   ("the") 
        
        2 → [0.76, 0.44, -0.98, 0.67]   ("cat") 

        5 → [-0.45, 0.87, 0.34, -0.21]  ("is") 

        4 → [-0.65, 0.38, -0.12, 0.79]  ("playing") 

    3. **Step 3**: The Model Learns Embeddings During Training

        Initially these embeddings are randomly initialized. During training, embeddings are adjusted based on the model’s task (e.g., predicting the next word in GPT). Similar words (e.g., "dog" and "cat") get similar embeddings.

        Embeddings are adjusted using backpropagation and gradient descent during training.
        
        1. **Step 1**: Input Text is Tokenized

            Example sentence: "The cat is playing". Tokenized: ["the", "cat", "is", "playing"]. Converted to token IDs: [1, 2, 5, 4].
        
        2. **Step 2**: Lookup Initial Embeddings from the Embedding Matrix
        
            1 → [0.12, -0.56, 0.89, 0.34]   ("the") 
            2 → [0.76, 0.44, -0.98, 0.67]   ("cat") 
            5 → [-0.45, 0.87, 0.34, -0.21]  ("is") 
            4 → [-0.65, 0.38, -0.12, 0.79]  ("playing") 

            These vectors are random before training.

        3. **Step 3**: The Model Predicts Output:

            If training a language model, it might predict the next word. If training a classifier, it might predict a category.

        4. **Step 4**: Compute Loss (How Wrong the Model Is):

            Example: The model predicts "playing" when the correct next word is "sleeping". The loss function (e.g., Cross-Entropy Loss) computes the error.

        5. **Step 5**: Backpropagation Updates Embeddings:

            The error is propagated back through the model. The embedding matrix is updated using gradient descent. Similar words get pulled closer together in vector space.

            Remember embeddings themselves are learnable parameters and therefore are seen as weights and are learned during training.

            Once the loss is calculated we do backpropagation
        
            1. The loss is calculated at the output layer of the model.
            2. The gradient of the loss is calculated with respect to all model parameters, including the embedding matrix (4 learnable parameters for our example). The gradient tells the model how to change its weights to reduce the loss (i.e., make better predictions). The gradients of the embeddings show how much each embedding vector should be adjusted.
            3. The embedding vectors are updated in the opposite direction of the gradient to minimize the loss. 

                embedding = embedding - learning_rate * gradient_of_embedding

        5. Step 6: Adjusted Embedding Vectors After Training

            "cat" $\rightarrow$	2	$\rightarrow$ [0.68, 0.42, -0.25, 0.14] 

            "dog" $\rightarrow$	5	$\rightarrow$ [0.70, 0.44, -0.26, 0.18] 

            "car" $\rightarrow$	8	$\rightarrow$ [0.12, -0.60, 0.80, -0.25] 


            Notice "cat" and "dog" became more similar, while "car" remains different.

    4. Why Do Similar Words Get Similar Embeddings?

        This happens due to contextual training. Words that appear in similar contexts get similar vector representations.        
    
Now that we have token embeddings, we need efficient ways to search, retrieve, and store them. This is where vector databases come in.

---

## Vector database

A vector database is a specialized database designed to store and search for high-dimensional vectors efficiently.

How Vector Databases Work with Embeddings

1. Step 1: Convert Words/Sentences into Vectors: We can store embeddings of words, sentences, or documents in a vector database.
2. Step 2: Perform Nearest Neighbor Search: If we query "intelligent", the vector database can return similar words like "smart", "clever", "wise". This is done using approximate nearest neighbor (ANN) algorithms like: FAISS (Facebook AI Similarity Search) ScaNN (Google) HNSW (Hierarchical Navigable Small World graphs).

---

## LLM architectures and training


###  manifold-constrained Hyper Connections

A deep neural transformer models can have multiple residual connections where we mix outputs of earlier layers to later layers. This assumes "All layers are safe and compatible to mix". However this may not be true and may confuse the model. mMHC only connects layers if they speak the same language by checking the compatibility first and if compatible then allow for connection otherwise weakens the connection. This is done by using post-mapping gates and are learned during training.

---

###  Mixed Precision Training

Most LLMs are trained using MPT since it can be expensive to use the fp32 parameters for both the forward pass and the backward pass which is done using scaling the parameters from fp32 to bf16. Which can lead to smaller memory based inference model based on bf16.


    1. Forward pass: uses bf16 (or sometimes fp16). This reduces memory usage and speeds up computation.
    1. Backward pass / gradients: stored in fp32. This preserves numerical stability during weight updates.
    1. This technique is often called mixed-precision training or automatic mixed precision (AMP).
    1. GPT models (OpenAI GPT-3, GPT-4, GPT-4-turbo) → mixed-precision (bf16/fp32).
    1. Other modern LLMs (LLaMA 2/3, MPT, Falcon, Claude) → all predominantly use bf16/fp32 or fp16/fp32 mixed precision.


---

###  Mixture of Experts (MoE)

MoE is a sparse architecture where the model is divided into multiple "experts" (sub-networks), and only a subset of these experts is activated for a given input. A gating mechanism decides which experts to use.

---

###  Retrieval-augmented generation (RAG)

RAG is a way to augment your LLM at inference time

---

## LangChain

---

## LLM reasoning

###  Chain of thought reasoning (CoT)

---

###  Hierachial reasoning Model (HRM)

---

## LLM model features/ techniques

---

###  Function calling

Instead of the model giving plain text, it produces a structured API call (JSON with function\_name + arguments). Making it possible for the model to reliably trigger external code/APIs.

eg:-
```{
  "name": "get_weather",
  "arguments": {
    "city": "London"
  }
}
```

---

###  Few shot function calling

You teach the model to perform function calls by showing a few examples of tool calls in the prompt. Open models (like gpt-oss-20B, LLaMA, Mistral, etc.) don’t always have built-in function calling support, so you simulate it with examples.

---

###  Structured outputs

Broader ML term, but in LLM context: It foreces the model to output valid structured data (JSON, XML, YAML) instead of free-form text. so that tools and downstream code can parse the output reliably. This can be done using libraries like Guardrails, Jsonformer, Outlines, OpenAI structured outputs

---

###  Tool use

In early “agents” research (AutoGPT, LangChain, etc.), tools are external functions or APIs the model can call. The LLM isn’t just generating text; it decides when to call tools (calculator, web search, DB query, filesystem, MCP server, etc.). MCP servers are just standardized tool providers. Tool use = the model invoking them.

---

## LLM Quantization

Quantization techniques are methods to reduce the precision of a neural network’s weights (and sometimes activations) to make models smaller, faster, and more memory-efficient, while trying to preserve accuracy.

---

###  Post-Training Quantization (PTQ)

**subsubsection:PTQ**

This quantizes a pre-trained model without retraining. Map floating-point values linearly to integers. 

1. Scale the weights: Find the min and max values of a weight tensor. where k = 4 for 4 bit quanization. This maps the continuous weight range to discrete levels. Compute a scale factor:
$$
scale = \frac{max - min}{2^k - 1}
$$   
2. Map weights to integers: Each FP16 weight is mapped to an integer between 0 and 15 (for 4-bit). The rounding method matters: naive rounding can introduce large errors; advanced methods like GPTQ or AWQ minimize error propagation.
$$
w_{int}=round(\frac{w-min}{scale})
$$
3. Store the scale separately: During inference, you reconstruct approximate FP16 weights: 
$$
w_{approx} = w_{int} \times scale + min
$$

**Examples: GPTQ, AWQ, GGUF, standard INT8 quantization.**

---

###  GPTQ (Generative Pre-trained Transformer Quantization)

1. Philosophy: (Quantize for Storage, De-quantize for Calculation) The goal is to store the model in as little VRAM as possible but perform the actual math using the GPU's native 16-bit floating-point capabilities. 
2. Hardware Target: GPU only.
3. Quantization process: A much more complex and slow process. It quantizes the model layer by layer. After quantizing a weight, it measures the error (the difference from the original value) and then adjusts the remaining, not-yet-quantized weights in the same layer to compensate for that error. This is the "fancy rounding" you mentioned. It's an optimization process that tries to minimize the overall error across the entire layer.
4. How it Runs: It Uses ultra-fast GPU kernels to de-quantize the 4-bit weights back to 16-bit floats on-the-fly, just in time for the calculation, and then performs the math in FP16.
    1. Layer-by-Layer Quantization with Calibration Data: It operates on the model one layer at a time. To do its job, it first needs to understand how a layer behaves. It does this by feeding a small amount of calibration data (a few hundred sample prompts) through the original, unquantized layer to observe the activations.
    $$
    \text{For i = 1 to N:}
    $$
    2. Block-Based Quantization (The Similarity): Just like GGUF, GPTQ groups the weights of a layer into small blocks (this is the group\ parameter, e.g., 128 numbers). It also calculates a scale for each block to define the quantization range.
    $$
    w_{i, int}=round(\frac{w_i-min}{scale})
    $$
    3. Sequential Optimization with Error Compensation: This is the heart of GPTQ. Instead of rounding all the numbers in a layer at once, it quantizes the weights sequentially within each layer and actively compensates for the errors it introduces. Meaning for all the error that is introduced by upadting the first block the model nudges (updates) the rest of the yet non-quantized weights to counteract the mistakes make by quantizting the first block. This happens for all the blocks. Where $H^-1$ is the inverse Hessian guiding the error distribution.  
    $$
    error_i = w_i - (w_{i, int} * scale)
    $$
    $$
    W_{remaining} = W_{remaining} + \text{distribute\_error} (error_i, H^{-1})
    $$

The GPTQ model have an extension of .safetensors and must be run on GPU. When the model runs, it completely ignores the complex quantization process. It simply performs the reverse operation on-the-fly

1. Take the stored 4-bit integer.
2. Multiply it by the stored scale.
3. This results in a 16-bit float.
4. Perform the matrix multiplication using the GPU's native, ultra-fast 16-bit Tensor Cores.

The GPTQ models can also be further fine-tuned using QLoRA.

---

###  GPT-Generated Unified Format (GGUF)

GGUF is not a quantization method. It is a file format created by the llama.cpp project to store LLMs efficiently. Quantization methods (like Q2, Q3, Q4, Q5, Q8) are different ways of compressing the weights inside a GGUF file. So GGUF = container and Q2/Q3/Q4/Q5/Q8 = how the weights inside that container are compressed. This technique is almost PTQ for reducing the weight's precision. With the only difference in its Philosophy being

1. Philosophy: Quantize for Integer Math. The goal is to create a model that can be run directly using fast integer operations, making it ideal for CPUs.
2. Hardware Target: CPU first, with GPU offloading as a bonus.
3. Quantization Process: A relatively simple and fast process of block-based scaling and rounding. The magic is in the clever structure (k-quants) and the highly optimized C++ code that runs it.
4. How it Runs: Performs math largely in the integer domain.
  
Here is how it works

1. Block-Based Quantization: The model's weights (which start as 32-bit or 16-bit floating-point numbers) are not quantized one by one. Instead, they are grouped into small "blocks" (e.g., a block of 32 or 256 numbers).
2. Find a Scaling Factor (Delta): For each block, the algorithm finds the number with the largest absolute value. This value is used to calculate a single, high-precision floating-point number called the scale or delta. This scale essentially defines the range or volume of all the numbers in that block.
$$
scale = \frac{max - min}{2^k - 1}
$$ 
3. Map to Integers: Every floating-point number in the block is then divided by this scale and rounded to the nearest integer that fits within the target bit-depth (e.g., for 4-bit, the integers can range from -8 to +7).
$$
w_{int}=round(\frac{w-min}{scale})
$$

1. A single, high-precision scale (a float).
2. A whole block of tiny integers (e.g., 32 numbers, each being 4-bit).

When the model runs, llama.cpp performs matrix multiplications by first unscaling the integers on-the-fly (integer * scale) or, more cleverly, by using highly optimized integer math that incorporates the scale factor directly.

The K in K-Quants: The Super-Block Innovation. The most advanced GGUF quants (the ones with \_K\_ in the name, like Q4\_K\_M) add another layer of intelligence. They use super-blocks. A super-block is a group of smaller blocks.

1. Instead of just one scale per block, they might use a more precise scale for the entire super-block and then smaller, less precise scales for the individual blocks within it.
2. This allows for a better distribution of precision, leading to higher model quality for a similar file size. It's a very clever trade-off.

GGUF models are CPU first models so when running a .gguf model you must set the n-gpu-layers setting in llama.cpp. It sets the number of model layers to load onto GPU. If you set the number to like 2 then 2 layers of GPU will be loaded onto the GPU and rest will work from CPU. However if you want the full model to run on GPU you set a very high number like 999 which will model the whole model to GPU. Also a GGUF file cannot be further fine-tuned.

---

###  Quantization-Aware Training (QAT)

The model learns to cope with low-bit weights during training. During forward pass: weights and activations are simulated as low-bit integers. However during backward pass: gradients are computed as if weights were still full precision (using Straight-Through Estimator (STE)). This leads to the model adjusting its weights to minimize the impact of quantization errors.

1. Start with a full-precision model: Base model is in FP32 or FP16. eg:- $w = 0.37$
2. Insert fake quantization operators: 
    1. Compute scale factor for quantization. for 4 bit quantization
    $$
    scale = \frac{max - min}{15} = \frac{1 - 0}{15} \approx 0.0667
    $$    
    2. Map to low-bit integers (simulate 8-bit, 4-bit, etc.).  
    $$
    w_{int} = round(0.37/0.0667) = round(5.55) = 6
    $$
    3. Map back to floating point for computation. so the actual weight in memory is still 0.37 (to which the gradient updates are applied) but forward pass is done on 0.4.
    $$
    w_{fake} = 6 \times 0.0667 \approx 0.4
    $$

This is called “fake quantization” because the actual weights remain full precision in memory, but the forward pass sees quantized values.

1. Forward pass: Input goes through the network. All operations use quantized weights and activations. Output approximates what the low-bit network would produce.
2. Backward pass (gradient computation): Gradients are computed with respect to full-precision weights, not the quantized values.
3. Update the full-precision weights using gradients. 
4. Repeat: forward pass simulates low-bit, backward pass updates FP32/FP16 weights.

After training, weights can be actually quantized for inference (INT8, INT4). Once model is trained we can do real quantization assume $w's$ final value is 0.37. 

$w = 0.37 \rightarrow$  round to 6 $\rightarrow$ store as 4-bit integer 6.

$$
w_{approx} =  6 \times 0.0667 \text{(scaling factor)} \approx 0.4
$$

---

###  Mixed Precision / Layer-wise Quantization

It is a variant of PTQ or QAT variant. Which uses different bit-widths for different layers or weight groups based on sensitivity.

---

###  Low-Bit Fine-Tuning Approaches

Hybrid quantization + fine-tuning. Famous methods are qLORA, LoRA + PTQ/GPTQ base, etc.


---

## LLM Fine-tuning

---

###  Full model fine-tuning

It updates all parameters of the pre-trained LLM on a new dataset.

---

###  PEFT (Parameter-Efficient Fine-Tuning)

General library for methods like LoRA, Prefix Tuning, Prompt Tuning, Adapters. Adapt LLMs with minimal additional parameters.

---

###  LoRA: Low-Rank Adaptation of Large Language Models

Consider $W_0$ model with params : 1000$\times$1000. We need to fine-tune this however in the traditional fine-tuning approach the number of parameters are way too high. So we use the technique of low-rank decomposition of matrixes to reduce the matrix size of the model's parameters making it $\Delta W$ so that fine-tuning time is reduced. 

Consider $\Delta W = A \times B$
where A is $A_{1000\times r}$ and B is $B_{r\times 1000}$

Here r is the rank of the matrix, which can be chosen to be much smaller (eg:- 4, 8, 16). So this makes the trianing parameters much smaller (m $\times$ r) + (r $\times$ n). In our case = (1000 $\times$ 4) + (4 $\times$ 1000) = 8000 < 1000000

However we don't specifically multiply AB to get back $\Delta W$. It works like this

1. $\Delta W = AB$
2. h = Wx (In the forward pass)
3. h = Wx + $\Delta Wx$ (modified by LORA)
4. $\Delta Wx = (AB)x \rightarrow A(Bx) \rightarrow$.
5. so now $z = Bx$ and $u=Az$ 
6. z = B $\times$ x : Here B is [r $\times$ 1000] and x is [1000 $\times$ 1] (this projects x into a low-rank bottleneck space of dimension r, i.e. z is of dim [r $\times$ 1]) 
7. u = A $\times$ z : Here A is [1000 $\times$ r]  and z is [r $\times$ 1]  resulting in  being of dim [1000 $\times$ 1]
8. W is frozen and its weight does not change
9. LORA only adds and extra branch that nudges the output using $\Delta W$

However for $\Delta Wx$ the magnitude can be too small or large compared to $Wx$ depending on how A and B are initialized. If it is too large the training blows up, however if it too small gradients vanishes and training is ineffective. Also if the rank is smaller the expressivity of $\Delta W$ can reduce (since very low number of params of say rank=2). This can cause scalability issues. We can counter this by using a scaling factor $\alpha /r$:

$$\Delta Wx = \frac{\alpha}{r}A(Bx)$$

1. r: is the rank (the size of the bottleneck)
2. $\alpha$: is the hyperparameter (8, 16, 32...)

---

###  QLoRA: Efficient Finetuning of Quantized LLMs

It can be used to fine-tune very large models (7B, 13B, 30B) on limited GPU memory.

The base model is quantized (usually 4-bit, sometimes 8-bit) to save GPU memory. Suppose you have a pre-trained model whose weights are in FP16 (16-bit floating point). This is done using a Post-Training Quantization (PTQ) method.

This reduces memory usage by4-8x. If you simply chopped off the lower bits of FP16, Large rounding errors would occur also The model performance would drop dramatically.

These weights are frozen — no gradient updates happen on the base model. Now we can add LoRA adapters (low-rank matrices (A and B) into certain layers) which are trained just like LORA where base model remains frozen. After which these adapters can be remain separate or merged with the base models.

---

###  Prefix Tuning

Learn continuous “prefix” embeddings prepended to the input sequence. Base model remains frozen.

---

###  Prompt Tuning

Learn a small set of task-specific embeddings used as prompts for the frozen model.

---

###  Adapter Tuning

Introduce small feed-forward modules (adapters) in the Transformer layers. Only train adapters, not the base model.

---

## LLM fine-tuning for alignment

---

###  SFT (Supervised Fine-Tuning)

Fine-tuning a pre-trained LLM on a curated dataset of input-output pairs (questions \& answers, instructions \& responses). Base model can be full-precision or frozen with LoRA/QLoRA adapters.

###  RLHF (Reinforcement Learning from Human Feedback)

It is a a three-step process to align model outputs with human preferences.


    1. Supervised Fine-Tuning (SFT)
    1. Reward Model (RM): Train a separate model to score outputs based on human preference data.
    1. Use RL (commonly PPO – Proximal Policy Optimization) to update the LLM so that it maximizes the reward score.


Make the model produce answers humans find helpful, safe, and aligned.

---

###  ICL / Instruction Tuning

Fine-tuning with large-scale instruction datasets so the model generalizes to unseen instructions. Often part of SFT, used before RLHF.

---

###  RLAIF (Reinforcement Learning from AI Feedback)

Instead of human feedback, another model generates the reward signal. Useful when human labeling is expensive.

---

###  Reward-Conditioned Fine-Tuning

Incorporates numerical reward values during fine-tuning, sometimes without RL.

---

## LLM layers

**Layer** | **Examples** 
|---|---|
Application Logic Layer | LangChain, LangGraph 
Model Serving / Inference | vLLM, llama.cpp, OpenAI API 
Model Training | PyTorch, DeepSpeed, FSDP 
Hardware | GPU, CPU, TPU 

---

## LLM inference engine

LLM inference engines are for running large LLM models locally. It does not train models.

---

###  llama.cpp

This is optimized for **CPU-first**, low memory and local execution.

1. Loads LLM weights 
Uses GGUF format, Can load 7B-70B models depending on RAM
2. Runs fast inference 
Highly optimized C++ kernels 
AVX / AVX2 / AVX-512 (CPU vectorization) 
OpenMP (multi-threading) 
Optional GPU offload (CUDA / Metal / Vulkan) 
3. Enables aggressive quantization 
16-bit $\rightarrow$ 8-bit $\rightarrow$ 4-bit $\rightarrow$ even 2-bit 
Example 7B FP16 $\approx$ 14 GB 
7B Q4 $\approx$ 4-5 GB
4. Provides API and tools. CLI chat, http server, python bindings, embedding generation and streaming tokens

The most important reason is that it is able to run LLMs without GPUs.

---

###  vLLM

This is optimized for **GPU-first}, low memory and local execution.


1. **Loads LLM weights**

    Uses HuggingFace / PyTorch model formats 
    Supports large models (7B--70B+) depending on GPU memory
2. **Runs high-throughput inference** 

    GPU-optimized CUDA kernels 
    PagedAttention for efficient KV-cache management 
    Designed for serving many concurrent requests
3. **Efficient memory utilization**

    KV-cache paging avoids GPU memory fragmentation 
    Enables longer context lengths and higher batch sizes 
    Scales well across multiple GPUs
4. **Production-grade serving engine** 

    OpenAI-compatible API server 
    Token streaming support 
    Optimized scheduling for multi-user workloads


The most important reason is that it enables extremely high-throughput and low-latency LLM inference on **GPUs only**, making it ideal for production LLM serving with many concurrent users.

---

###  SGLang

---

###  TensorRT-LLM

Same but is Nvidia-only and supports CUDA

---

## LLM inference and benchmarks

---

###  Configuration parameters for LLM inference



1. **Question:** what is Probability threshold for top-p sampling?

    **Answer:** 

    When a language model generates text, it predicts a probability distribution over all possible next tokens. Some tokens are more likely than others. Top-p sampling doesn’t just pick the single most probable token, instead
    
    1. It sorts tokens by probability from highest to lowest.
    2. It then selects the smallest set of tokens whose cumulative probability \geq p (the probability threshold).
    3. It samples randomly from this set, rather than always picking the top one.
    

1. **Question:** what is temperature? 

    **Answer:** LLM predicts the probability distribution of the next token. Temperature controls how “confident” or “random” the model is when picking the next token. Probability is adjusted like this
    $$
    P^{`}_{i} = \frac{e^{\frac{log P_i}{T}}}{\sum_j e^{\frac{log P_j}{T}} }
    $$
    Where T is the temperature, i is the probability of the token that is being considered and j is the probability of set of all possible tokens. How it works
    
    1. Temperature = 1: No change, use the original probabilities.
    2. Temperature < 1 (e.g., 0.5): The distribution becomes sharper, making high-probability tokens more likely. Output is more deterministic / predictable.
    3. Temperature > 1 (e.g., 2.0): The distribution becomes flatter, giving low-probability tokens more chance. Output is more random / creative / surprising.

---

## Model Context Protocol (MCP) for LLMs


## Powering LLMs with AI Agentic workflows

## Building LLM powered applications
