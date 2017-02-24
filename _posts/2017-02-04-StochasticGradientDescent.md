---
layout: post
title: Stochastic Gradient Descent recomendations
published: false
---
If you compute the gradient on the half of your training data and also on the other half of the training data, you will always get the same answer on both of them, is better to compute the gradient on the first half, then update the weights and then compute the gradient on the second half.
Typically we use a mini batch size of 10 or 100 examples or 1000 examples.
So we can conclude that ** Is not optimal to compute the gradient for your weights on the entire training set.**

**Online Learning**
The extreme version of the mini-batch learning approach is called online learning, where you compute the gradient for just one training case, update the weights and then keep with another training case.
In general, the Online Learning is quite extreme for most applications so is better to use the mini-batches most of the time.

**Full batch learning advantages**
One advantage of full batch learning is that less computation is used for weight updates. Another advantage of batch learning is that you can compute the gradient for a whole bunch of cases in parallel because of the nature of matrix multiplication.
Conjugate gradient is a full-batch technique that can be very efficient too.
  
**Randomize your data**
On thing to have into account when doing full batch or mini-batch learning is that for each batch, the training cases must be very different with respect to their label, if a group batch have most of their training cases of the same type this will unnecessarily update the weights in a bad way and the training will not be so efficient at the end, to be able to have a very unique batch just apply a random function over all your data before training.

    
**Basic Mini-Batch gradient descent algorithm**

* The basic steps for doing a basic mini-batch gradient descent:
    * Guess the initial learning rate
    * If the error keeps getting worse or oscillates wildly, reduce the learning rate.
    * If the error is falling fairly consistent but SLOW, increase the learning rate.
    * Towards the end of minibatch learning it always helps to turn down the learning rate, this removes fluctuations in the final weights caused by the variations between mini batches, AND you want a fine set of weights that is good for many mini batches.
    * The behavior mentioned above can be automated in a simple program
    * A good time to turn down the learning rate is when the error stops decreasing consistently, and good criteria to affirm that the error stops decreasing consistently is to use the ERROR metric on a separate VALIDATION SET(this is a bunch of examples that are not used for training nor for the test set)
