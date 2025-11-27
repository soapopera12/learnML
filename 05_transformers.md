# 05_transformers.md
# 5 Transformers

## 5.1 T5
The T5 model (short for Text-to-Text Transfer Transformer) showed that scaling up transformers (hundreds of millions to billions of parameters) and pretraining them on massive data (C4 dataset) gave strong generalization. Instead of BERT’s masked language modeling (masking individual tokens), T5 used span corruption:
* Random spans of text are replaced with a unique sentinel token (like `<extra_id_0>`).
* The model learns to generate the missing spans.
* Example:
  - Input → "The `<extra_id_0>` is on the `<extra_id_1>`."
  - Output → "`<extra_id_0>` book `<extra_id_1>` table"

It can be used for Summarization, translation, classification, QA, reasoning etc. Chatmodels like chatGPT require long form generation and dialogue memory, however T5 models are less chatty and harder to scale at 100B+ params.