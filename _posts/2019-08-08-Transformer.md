---
layout: post
title: Transformers
published: true
---

Since the breakthrough paper [https://arxiv.org/abs/1706.03762], Transformers had been achieving state of the art in many tasks, here I explain in easy terms what a transformer is.

A transformer model is a neural network model that uses the following approaches:

- It uses a form of direct access memory called Attention, which helps the network access the inputs at any time in the inference process, the network learns to focus on the most relevant inputs in the right context.
- The original transformer uses the seq2seq approach using an encoder module and a decoder module with the main task of machine translation, the encoder encodes all the inputs to a hidden state, this hidden state is special because is the result of encoding learned representations according to the task in hand, this hidden state is created with the help of attention layers. The decoder module takes the entire hidden representation of the input and with repeated inference calls, it starts to output a sequence of outputs one-word token at a time, for example, it starts to output the input sequence translated to another language word by word.
- The encoder uses multiple attention layers (Multi-Head Attention) to focus on the inputs from different perspectives, for example, a word can be viewed in terms of genre or terms of quantity, in the end, all the attention heads are combined into a single representation.
- The decoder uses multiple masked attention layers (masked multi-head attention), it is masked because we want the decoder to learn its attention only from the context of the seen words so far (it can't see the future), this makes the original transformer model unidirectional. Some tasks may require bidirectional knowledge of the inputs (for example Question/Answering) for this reason variation of transformer architectures had emerged (see BERT) where the masking is not made, so the attention has access to the entire input at all times.
- The decoder also uses multi-head attention to focus on the outputs from the encoder.
- 
The transformer architecture consists of multiple transformer modules stacked, usually, there are multiple types of transformers, encoder specialized, decoder specialized, or both, the nice thing about transformers is that they are highly parallelizable, also they provide multiple attention layers, this means that they learn to selectively process what input to take from the previous layer, this includes the input layer (self-attention) to the hidden layers, this is powerful because the transformer learns its architectural connections according to the learned task.

## Q, K, V matrices
The way for the transformer to do attention to each layer is via the Q K V matrices, The Q (Query) matrix contains the embeddings from the previous layer this is the data that we are going to query, the K(key) matrix contains the keys that will be used over the query matrix, this multiplication causes some query values to have higher values, the higher values will be the ones that will have more relevance to the next layer, but in reality, we don't pass the higher query values but the corresponding V values. This is essentially a dictionary lookup.

## Multi-head attention
The transformer has multi-head attention, this is multiple attention learned weights all occurring in parallel, in other words, the transformer learns multiple ways to give importance to the inputs and hidden inputs, all of this transformer attentions run in parallel but at the end of the layer they are combined, this can be seen as a kind of ensemble that provides good regularization and hence good generalization.

# Advanced Transformers
VilBERT tries to combine language tasks and vision tasks all in one architecture, this model achieves this by using Co-Attention layers, these layers are layers that receive input streams according to a specific domain, language, or vision, but they share their attention with the other specialized stream, in this way, the language stream attention layer learns to focus its attention on the vision task, and the vision stream attention layer learns to focus its attention on the language task.

Transformers has superior performance compared to previous attention based architectures via:
-	Multi query hidden state propagation (self-attention)
-	Multi-head attention (multiple attention mechanism for the same item that is combined)
-	Residual connections (layer norm)
