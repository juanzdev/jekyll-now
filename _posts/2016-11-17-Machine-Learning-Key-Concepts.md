---
layout: post
title: An easy introduction to Machine Learning Part 2
published: true
---

## Generalization
When you train a machine learning algorithm we know that the input data is changing the internal values of the model, you can train a machine learning algorithm with you data so well that the model can produce an exact output function that resembles exactly as the training set, this can be risky because your goal is to be able to identify the hidden rule that describes your data, if your model can deduce this hidden rule then the model will be able to behave very well on unseen data, the capability of the model to deduce this hidden rule is what is called generalization, because your model was trained on the training set but was able to learn the rules that describe the data very well, the model generalizes well.

## Overfitting
When you train a machine learning algorithm with small amounts of data or maybe with bad data your model after training will try to describe your training set exactly as it is, this is bad because your model must be prepared to see unseen data on real-world problems, in this case, your model is so tight to the training set that at the end is unpractical to use it on new data, a common way to overcome overfitting is to gather much more data or ease the learned function. 

## Underfitting
When you want to train a machine learning model with some data, and this data is kind of complex (maybe is too much data or the hidden rule is too complex to learn) and you choose a simple learning model it can happen that the model, although we have lots of data, is incapable of learning the complex relationships that describe the data, in this case, is too hard for the chosen model to come up with a result function, a common way to overcome underfitting is to change the chosen model with more complex ones.

## Parameter
A parameter is a changing value of a statistical learning model, the parameter will be adjusted with greater values or lesser values on the training process, at the end when you have a machine learning model ready for production it means the configured parameters represents the final and optimal solution that you were looking for.

## Hyperparameter
An hyperparameter is a configuration value that you can make to a statistical machine learning model, in machine learning there are a bunch of statistical models, these models have also some configurations, for example, one statistical model is neural networks, an hyperparameter, in this case, could be the number of neurons that it has per layer or the total number of layers.

## High Bias/Low Bias High Variance/ Low Variance
When you train a model with data there will be always an uncertainty that there will be some kind of error, no model is perfect in machine learning, if a model represents the data so well that in overfits the data we have a high variance problem, when our model is not able to represent the model but just a weak approximation we have a high bias problem, this tradeoff must be regulated in machine learning to come up with the best results in ML.




