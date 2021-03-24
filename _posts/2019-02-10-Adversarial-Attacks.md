---
layout: post
title: Adversarial Attacks
published: true
---


Attention is essentially weights that depend on the input, this weights can depend on the input of the input layer (self attention) or input that is the result from a hidden layer (attention), attention mechanisms are very powerfull because we dont have to bias the network architecture to the way we think the network will perform tha task, for example CNNs are network architectures that are specialized to process 2D or 3D dimensional data, the convolution operations process the input in a specific way, the neurons process the inputs via their corresponding receptive fields, all of this machinery is architectural bias, because the experts think the network will perform better if the network process the input this way. But what if there is a better way to process 2D or 3D inputs, what if we let the model learn by itself how to process the inputs according to some task? Enter the Attention layers.

# Types of Attacks

- single pixel attacks
- white box attacks (the attacker needs to know the model architecture)
- black box attacks (the attacker only needs to know the ouputs of the model (how it behaves))
