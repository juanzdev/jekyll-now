---
layout: post
title: Prediction vs Inference in Machine Learning
published: true
---

Having a clear understanding about these two concepts is very important to undertand a lot of the concepts in machine learning.

##Prediction
The most intricated algorithms in machine learning for example those associated with deep learning are very advanced algorithms that usually ingest a lof of data to create inner patterns, for example a functional neural network will need a lot of data if it wants to be highly reliable, when you train a neural network with backpropagation a lot of things are happening inside the structure of the neural network but you dont know exactly what is happing, what we know is that the weights are being balanced and corrected with each new input, the final neural network configuration can be associated with a mathematical function that can describe the base original input data with the associated outputs, this functions is a black box because we dont know the implementation, but in some cases we dont care about the implementation because whaty we want is that the function outputs a very good result. In this case we are saying the neural network is trained and can predict most of the values we provide it. It is called prediction because we dont have knowledge what kind of rules produce the output value of the neural network, for us this is a black box that outputs a good value most of the times, most of the deep learning algorithms out there output predictions because of their complexity but at the end we are fine with that.

##Inference
On the other side it is good to understand why a machine learning algorithm is giving us a especific result, this is to be able to understand the relationship between the multiple variables that our input data have in common, this is great because we can put algorithms to work for us to understand the data, and even better we can have insights of the relationship between one variable or another. The reality is that most of the complex and most powerfull algorithms will not give us inference about the data because they handle a lot of input parameters that becomes very complex to understand using statistical tools, this is unfurtunate because it will be great to have a generated rule about an autonomus car driving, on the other side is util for other kind of problems with fewer parameters.

