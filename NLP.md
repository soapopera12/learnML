# Natural Language processing

## Basic Text Processing

### Tokenization (word, subword, sentence)

Types of tokenizers
1. Word tokenizers
2. Subword tokenizers
3. Sentence tokenizers

**Word Tokenization**

**Definition**

Word tokenization splits text into individual words (and often punctuation) as basic units.

**Example**

1. Input: I love NLP!
2. Output: ['I', 'love', 'NLP', '!']

**Famous Algorithms and Usage**

1. Whitespace tokenization: simple baseline for clean text
3. Rule-based tokenizers (NLTK, spaCy): used in classical NLP pipelines
4. Usage: POS tagging, parsing, traditional NLP tasks

---

**Subword Tokenization**

**Definition**

This is widely used today. Subword tokenization splits words into smaller meaningful units such as prefixes, roots, or suffixes.

**Example** 

1. Input: 'unhappiness'
2. Output: ['un', 'happi', 'ness']

**Famous Algorithms and Usage**

1. BPE (Byte Pair Encoding): used in GPT models
2. WordPiece: used in BERT
3. SentencePiece: used in T5 and multilingual models
4. Usage: standard in transformer-based NLP

**Subword Tokenization Methods and sub-methods**

1. **Frequency-Based Subword Tokenization**

    **Definition:** 

    Builds subwords by merging the most frequently occurring character/token pairs in the data.
    **Tokenizers:**

    1. Byte Pair Encoding (BPE)
    2. Byte-Level BPE (still subword, just at byte granularity)
    3. SentencePiece (BPE variant)
    4. GPT-style BPE (e.g., tiktoken)

2. **Probability / Likelihood-Based Subword Tokenization** 

    **Definition:**

    Learns subwords by maximizing the likelihood of the data, selecting tokens based on probabilistic scoring instead of raw frequency.

    **Tokenizers:**

    1. WordPiece
    2. Unigram Language Model
    3. SentencePiece (Unigram LM variant)

---

**Sentence Tokenization**

**Definition**
Sentence tokenization splits text into individual sentences based on punctuation and linguistic rules.

**Example**

1. Input: 'I love NLP. It is amazing!'
2. Output: ['I love NLP.', 'It is amazing!']

**Famous Algorithms and Usage**  

1. Punkt tokenizer (NLTK): rule-based sentence splitter
2. Statistical and neural sentence boundary detection
3. Usage: summarization, document processing, sentence-level classification

---

### Stopword Removal

**Definition**  

Stopword removal is the process of removing very common words that occur frequently in text but carry limited semantic meaning.

**What they are**  

1. Words that appear very frequently in a language
2. Examples include: 'the', 'is', 'and', 'in', 'of', 'to'
3. These words usually contribute more to grammar than meaning

**Example**

1. Input: 'I am going to the store and I will buy food'
2. Output: 'going store buy food'

**Famous Approaches, Usage, and When Not Used** 

1. Predefined stopword lists (NLTK, spaCy)
2. Not used in transformer-based models such as BERT and GPT

---

### Stemming (Porter, Snowball)

**Definition**

Stemming is the process of cutting words down to a base form called a stem using crude, rule-based methods. The resulting form does not need to be a valid word.

**What it is**  

1. A rule-based technique for reducing words to a common base
2. Groups similar word forms into a single representation
3. Does not use a dictionary and may produce non-real words

**Example**

1. Input words: 'playing', 'played', 'plays'
2. Output: 'play', 'play', 'play'
3. Input words: 'studies', 'studying', 'studied'
4. Output: 'studi', 'studi', 'studi'
5. Note: 'studi' is not a real word and that is acceptable in stemming

**How it works** 

1. Removes suffixes using predefined rules
2. Examples of rules:
    1. remove 'ing'
    2. remove 'ed'
    3. replace 'ies' with 'i'
1. Example transformations:
    1. 'studies' $\rightarrow$ 'studi'
    2. 'running' $\rightarrow$ 'run'
    3. 'happiness' $\rightarrow$ 'happi'
3. No dictionary lookup is used, only pattern-based transformations

**Famous Algorithms and Usage**

1. Porter Stemmer:
    1. Classic algorithm based on a sequence of suffix rules
    2. Example rules: 'ing' removal, 'ed' removal, 'ational' $\rightarrow$ 'ate'
    3. Example: 'relational' $\rightarrow$ 'relate', 'running' $\rightarrow$ 'run'

2. Snowball Stemmer:
    1. Improved version of Porter
    2. More consistent and supports multiple languages
    3. Slightly more accurate in practice
    4. Example: 'studies' $\rightarrow$ 'studi'

---

### Lemmatization

**Definition**

Lemmatization is the process of converting a word to its base dictionary form called a lemma using linguistic knowledge such as dictionary lookup and grammar rules. It follows a dictionary + grammar based method instead of strick rule based method used in stemming and produces valid words.

**What it is**

1. A normalization technique that maps different word forms to a meaningful base form
2. Uses morphological analysis and Part-of-Speech information
3. Ensures that the output is a valid word

**Example**  

1. Input words: 'running', 'ran', 'runs'
2. Output: 'run', 'run', 'run'
3. Input words: 'better', 'studies'
4. Output: 'good', 'study'
5. Output words are valid and preserve meaning

**How it works**

1. Uses dictionary lookup to find base forms
2. Applies grammatical rules
3. Uses Part-of-Speech tagging to determine correct lemma
4. Context is important for correct transformation
5. Example:
    1. 'better' as adjective $\rightarrow$ 'good'
    2. 'saw' as verb $\rightarrow$ 'see'
    3. 'saw' as noun $\rightarrow$ 'saw'

**Types / Approaches**

1. Rule + Dictionary based:
    1. Uses vocabulary and grammar rules
    2. Example: 'am', 'is', 'are' $\rightarrow$ 'be'
2. POS-aware Lemmatization:
    1. Uses Part-of-Speech tagging before lemmatization
    2. Improves correctness by considering context
    3. Example: 'saw' (verb) $\rightarrow$ 'see', 'saw' (noun) $\rightarrow$ 'saw'

**Famous Approaches and Usage**

1. WordNet Lemmatizer:
    1. Uses lexical database for English
    2. Common in NLTK
2. spaCy Lemmatizer:
    1. Uses rule-based and statistical methods
    2. Efficient and widely used in pipelines

---

### Handling Out-of-Vocabulary (OOV) Words

**Definition**

Out-of-Vocabulary (OOV) words are words that a model has **never seen during training**, and therefore has no representation for.

**Example**

**Training Vocabulary:**

1. cat, dog, run

**New Sentence:**

1. tiger runs fast

**OOV Words:**

1. tiger
2. runs (if only ``run'' was seen)

**Why is OOV a Problem?**

1. Model has no representation for unseen words
1. Cannot assign meaning to them
1. Leads to poor predictions


**Methods to Handle OOV**

1. **Unknown Token**

    Replace unseen words with a special token <UNK>.

    **Example:**

    1. tiger runs $\rightarrow$ <UNK> <UNK>

2. **Subword Tokenization (Modern Approach)**

    Break words into smaller known units.

    **Example:**

    1. tiger $\rightarrow$ tig + er
    2. running $\rightarrow$ run + ning

    **Common Methods:**

    1. Byte Pair Encoding (BPE)
    2. WordPiece

3. **Character-Level Models**

    Process text at the character level.

    **Example:**

    1. tiger $\rightarrow$ t, i, g, e, r

4. **Stemming / Lemmatization**

    Reduce words to their base form.

    **Example:**

    1. running $\rightarrow$ run

    **Where Each is Used**

    1. Classical NLP: <UNK>, stemming, lemmatization
    1. Modern NLP: Subword tokenization (BPE, WordPiece)

**When is OOV Less of a Problem?**

In modern Transformer-based models, OOV is rarely an issue because:

1. Subword tokenization can represent unseen words
1. Models can generalize from known subword units

---

## Language Modeling

Classical
N-grams
Markov assumption
Smoothing:
Laplace smoothing
Good-Turing
Kneser-Ney (important)
Neural Language Models
Feedforward NN LM
RNN, LSTM, GRU
Sequence modeling
Modern
Transformer-based LMs:
Autoregressive (GPT-style)
Masked LM (BERT-style)

---

## Morphology, Parts of Speech Tagging

Morphology
Morphemes (root, prefix, suffix)
Inflection vs derivation
POS Tagging
Rule-based tagging
HMM-based tagging
CRF-based tagging
Neural POS tagging (BiLSTM, Transformers)

---

## Syntax: structure of sentences

Constituency Parsing
Context-Free Grammars (CFG)
Probabilistic CFGs (PCFG)
Dependency Parsing
Dependency trees
Transition-based parsing
Graph-based parsing
Concepts
Parse trees
Ambiguity
Syntax vs semantics distinction

---

## Semantics (Meaning)



Lexical Semantics
Word meaning
Synonymy, antonymy
WordNet
 Distributional Semantics

Word2Vec:
CBOW
Skip-gram
GloVe
FastText

Concepts:

Embedding space
Cosine similarity
Analogies (king - man + woman ≈ queen)
 Compositional Semantics
Meaning of phrases/sentences
Simple models → RNNs → Transformers
 Word Sense Disambiguation (WSD)
One word, multiple meanings
Knowledge-based vs ML approaches
 Contextual Embeddings (Modern NLP)
ELMo
BERT
GPT embeddings

Words have different meanings in different contexts

---

## Information Extraction / Applications

This is where NLP becomes useful.

Core Tasks
Named Entity Recognition (NER) ✅
Identify entities (Person, Org, Location)
Models:
CRF
BiLSTM-CRF
Transformers
Relation Extraction
Find relationships between entities
Coreference Resolution
“Elon Musk... he…” → same entity
Classification Tasks
Sentiment Analysis
Text Classification
Spam detection
Generation Tasks
Machine Translation
Text Summarization
Question Answering
Dialogue systems (chatbots)
Retrieval \& Search
Information Retrieval (TF-IDF, BM25)
Semantic search

---

## Evaluation

Accuracy, Precision, Recall, F1
BLEU (translation)
ROUGE (summarization)

---

## Named Entity Recognition (NER)

Named Entity Recognition (NER) is a Natural Language Processing (NLP) task that identifies and classifies key information (entities) in text into predefined categories.

**Goal**

Extract structured information from unstructured text.

**Common Entity Types**

1. Person (e.g., Elon Musk)
2. Organization (e.g., Google)
3. Location (e.g., India)
4. Date/Time (e.g., 2024, Monday)
5. Money, Percent, etc.

**Example**

Sentence:

1. Elon Musk founded SpaceX in 2002 in the USA.

NER Output:

1. Elon Musk $\rightarrow$ Person
2. SpaceX $\rightarrow$ Organization
3. 2002 $\rightarrow$ Date
4. USA $\rightarrow$ Location

**How it Works**

NER is typically treated as a sequence labeling problem.

Each word is assigned a tag:

1. B-ENT: Beginning of entity
2. I-ENT: Inside entity
3. O: Outside entity

**Example:**

1. Elon (B-PER), Musk (I-PER), founded (O), SpaceX (B-ORG)


