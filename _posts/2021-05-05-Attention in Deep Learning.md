---
layout: post
title: Attention in Deep Learning
published: true
---

In deep learning, backpropagation constrain the learning to be smooth due to differentation. Deep learning models can learn hierarchical representations automatically given the data, all this process takes place by incrementally tuning the weights in a neural network given the loss function of interest. Attention is the mechanism to introduce a dictionary-like structure to the nerual network learning process. 

An Attention Layer \[Bahdanau et al. 2015\] can be seen as a differentiable dictionary, which provides valuable information to the neural network given a query state.

The problem with very deep models like recurrent neural networks is that they are so deep that the information encoded at some layer could be very far from another layer that needs the stored representation. Attention solves this.

Attention makes the information instantly available at any time through a deep model.

Attention is also learnable, this is, an attention mechanism in a neural network can be implemented using an attention layer.

The attention layer can be seen as a dictionary structure, remember that a dictionary is a data structure that is very fast at reading time, this is we provide a lookup to the dictionary with a unique key and the dictionary provides a value.

Attention in deep learning provides this dictionary structure in a learnable way (eg: can be learned with backpropagation)

At inference time for example, an attention layer can be useful when a neural network at layer 13th needs certain information from the original input. A deep model without attention would have to encode this input information throughout all the intermediate layers, and if the model is too deep the relevant information could be lost or get too noisy when updating the gradients (catastrophic forgetting). A deep model with attention can make a query(Q) lookup at any time to the differentiable dictionary, this differentiable dictionary provides the relevant information given the query, this query is not an exact query concept but is the query information that the neural network needs to perform given its actual state. Given the state query, the dictionary will return the most probable keys and their corresponding values that match the query context. The keys and values are subsets of information from the input. This dictionary is not provided by any means at training time, it is initially a set of random parameters, but at training time the dictionary will be learned based on the downstream task of interest. So the keys, values and the query representations that this dictionary expects will be learned at training time.

