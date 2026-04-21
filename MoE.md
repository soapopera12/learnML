# Mixture of Experts (MoE)

MoE trains many parameters, but only a few at a time. This neural network architecture consists of an encoder, experts and router (gating network). This architecture consists of many expert neural networks each specialized for different input type. During inference the model automatically selects the best top-k experts which help in reducing compute and inference time (since the entire model does not need to run for single input only topk experts).

Working of a MoE model:

1. The input passes through an encoder.
2. After this the input is passes through a router which selects the top-k experts.
3. Forward pass runs only through selected experts.
4. Loss propagates through router, encoder and the selected experts only.
5. The router is a small single layer neural network that has output equal to the number of experts in the model and therefore gives probability values of which experts to select in descending order.
6. While testing a new sample the router computes scores to select the top-k experts.

Experts specialize because they see different data.