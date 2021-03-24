---
layout: post
title: Transformers
published: true
---

The tranformer architecture consist of multiple transformer modules stacked, usually there are multiple types of transformers, encoder specialized, decoer specialized or both, the nice thing about transfrormers is that they are highly parallelizable, also they provide multiple attention layers, this means that they learn to selectevely process what input to take from the previous layer, this includes the input layer (self attention) all the way to the hidden layers, this is powerfull beause the transformer learns its own architectural connections according to the learned task.

## Q, K, V matrices
The way for the transformer to do attention at each leayer is via the Q K V matrices, The Q (Query) matrix contains the embeedings from the previous layer this is the data that we are going to query, the K(key) matrix contains the keys that will  be used over the query matrix, this multiplication causes some query values to have higher values, the higher values will be the ones that will have more relevance to the next layer, but in reality we dont pass the higher query values but the corresponding V values. This is essentailly a dictionary lookup.

## Multi head attention
The transformer has mult head attention, this is multiple attention learned weights all occuring in paralel, in other words, the transformer learns multiple ways to give inportance to the inputs and hidden inputs, all of this transformer attentions run in parallel but at the end of the layer they are combined, this can be seen as a kind of ensemble that provides good regularization and hence good generalization.

# Advanced Tranformers
VilBERT tryes to combine language tasks and vision tasks all in one architecture, this model achieve this by using Co-Attention layers, this layers are layers that receive input streams according to a specific domain, language or vision, but they share they attention with the other specialized stream, in this way, the language stream attention layer learns to focus its attention on the vision task, and the vision stream attention layer learns to focus its attention on the language task.

Transformers has superior performance compared to previous attention based architerctures via:
-	Multi query hidden state propagation (self attention)
-	Multi head attention (multiple attention mechanism for the same item that are combined)
-	Residual connections (layer norm)



