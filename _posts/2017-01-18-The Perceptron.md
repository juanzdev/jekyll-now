---
layout: post
title: The Perceptron
published: true
---

In Machine Learning to know the history and the inner workings of the perceptrons is important because it could be said that is the first architecture that tries to learn patterns with inputs of data.

The perceptron is a network that receives multiples fixed inputs of data specifically numbers, those inputs are connected to a special neuron by a set of synapses or weighs, the neuron will calculate a weighed sum over the inputs and the wieghs and will perform an activation function in this case a binary one.

The mathematical definition of the perceptron goes like this:
**formula**

In machine learning we call this functions score functions, because they gives us some unique number from multiple inputs they gives us a score, in this case because our neuron is a binary neuron it will gives us 1 or 0. 


**diagram of basic example of perceptron with positive and negative inputs and the output as a single bit**

As you can see we have negative and positive inputs and our perceptron will gives us an arbitrary binary number.

Now for example we want our perceptron to outputs the following states:


| Input 1   |      Input 2      |  Output |
|----------|:-------------:|------:|
| 1 |  1 |1 |
| 1 |   0  |   0 |
| 0 | 1 |    0|
| 0 | 0 |    0|


With our current configuration the output is completelly off, the only way we can tune our perceptron is by start changing the weights, the weights is the only variable that we can change.
We can start changing weights and start trying if the desired output match exactly with the objective ouput. lets try one time changing the weighs:

**new set of weights**

we can see that the result is not our expected ouput, we can keep trying but this will be too cumbersome and there is a better way to achieve this and that is learning the weights, to be able to do that we need aside from our score function a loss function.

**Loss Function**
You will start seeing this pattern a lot on machine learning models, we have a score function accompanied by a loss function, both are necessary for the learning to happen as we will see. For now lets concentrate on what a loss function is.

The loss function will tell us how bad or good our **score** function is behaving, as simply as that. There are a lot of loss functions from simply ones to more complex, here we are going to see the simplest loss function in Machine Learning the square diferences.



**The Perceptron Learning Procedure**

