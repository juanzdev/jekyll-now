---
layout: post
title: Neural Language Model with Caffe
published: false
---

In this blog post I will explain how you can implement a Word Embeddings implementation in Caffe using Bengio Neural Model and based on Hinton Coursera course code in Octave.


A neural network is compossed of what is called hidden layers, the state of this layers will be unknown, they are very useful because they are a representation of our data in different dimensions, this is the same as saying that the hidden layers are transformation of our features, this is commonly called different feature representations.

By working with different feature representations is possible to create very abstract associations of the data, for example when training a classifier of pictures we can create boundaries in high dimensional space to group "concepts" of pictures and to be able to differentiate how similar or diferent a picture is to other picture, this is possible because in those high dimensions the concept of a picture can be represented with a bunch of number combinations in a high dimensional data space.

## Word Embeddings
A high dimensional representation of a word is called a word embed, by having a different feature representation of the words we can group words by similarity not in syntax but in context.

Bengio was the first to propose an neural architecture for the word embedding problem:

-image of bengio neural net architecture-

This net works by the premisse that you have a static vocabulary with word codes(250 codes in total), this is, for each word we are going to assign it a unique code, in this case an integer. Then a look-up operation will take place over the embeed matrix, this means that initially this matrix will have random values but after back propagation each row of this matrix will represent the different and expanded feature representation of each word, this is the reason why this matrix is 250 by 50, in this case the number 50 is the expanded representation of each word, this is an hyperparameter and you can try with different values.


