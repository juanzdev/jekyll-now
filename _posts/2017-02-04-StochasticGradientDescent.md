---
layout: post
title: Stochastic Gradient Descent recomendations
published: true
---

Here are some recommendations when using Gradient Descent

Is not optimal to compute the gradient for your weights on the entire training set
    * If you compute the gradient on the half of your training data and also on the other half of the training data, you will always get the same answer, so it's not necessary to do that, instead you should compute the gradient on the first half, then update the weights then compute the gradient on the second half
    * The extreme version of this approach is called Online Learning, where you compute the gradient for just a training case then update the weights and keep with another training case
    * In general, the Online Learning is quite extreme and we don't want to go that far, is better and more used to use a batch learning approach with small mini-batches typically 10 or 100 examples or even 1000 examples
    * One advantage of batch learning is that less computation is used for weight updates
    * Another advantage of batch learning is that you can compute the gradient for a whole bunch of cases in parallel because of the nature of matrix multiplication, it is pretty fast.
    * On thing to have into account when doing batch learning is that for each batch, the training cases must be very different with respect to their label, if a batch will have the same answer(label) that will unnecessarily slosh the weights and the training will not be so efficient, to be able to have very characteristic batches (unique) is just to simply apply a random function over your sorting of training data.
    * So there are actually two methods for learning (full batch vs mini-batch), for full batch you can make it more optimal by applying conjugate gradient and other numerous methods created by the optimization community
    
Basic Mini-Batch gradient descent algorithm
    • Guess the initial learning rate
    • If the error keeps getting worse or oscillates wildly, reduce the learning rate.
    • If the error is falling fairly consistent but SLOW, increase the learning rate.
    • Towards the end of minibatch learning it always helps to turn down the learning rate, this removes fluctuations in the final weights caused by the variations between mini batches, AND you want a fine set of weights that is good for many mini batches.
    • The behavior mentioned above can be automated in a simple program
A good time to turn down the learning rate is when the error stops decreasing consistently, and good criteria to affirm that the error stops decreasing consistently is to use the ERROR metric on a separate VALIDATION SET(this is a bunch of examples that are not used for training nor for the test set)
