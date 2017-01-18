---
layout: post
title: The Perceptron
published: true
---

In Machine Learning the first version of neural networks are related to perceptrons and is quite important to understand them to grasp some important concepts that will be too common even on the latest ML topics.

The perceptron is a network that receives multiples fixed inputs of data specifically numbers, those inputs are connected to a special neuron by a set of synapses or weighs, the neuron will calculate a weighted sum of the inputs and the weights and will perform an activation function, in this case, a binary one.

The mathematical definition of the perceptron neuron or binary threshold neuron is defined as:

{% raw %}
$$
f(x) = \begin{cases}1 & Wx+b \gt 0 \\ 0 & \text{otherwise} \end{cases}
$$
{% endraw %}

In machine learning, we call these functions score functions because they give us some unique number from multiple inputs they give us a score, in this case, because our neuron is a binary neuron it will give us 1 or 0. 


**diagram of basic example of perceptron with positive and negative inputs and the output as a single bit**

As you can see we have negative and positive inputs and our perceptron will give us an arbitrary binary number.

Now for example we want our perceptron to outputs the following states:


| Input 1   |      Input 2      |  Output |
|----------|:-------------:|------:|
| 1 |  1 |1 |
| 1 |   0  |   0 |
| 0 | 1 |    0|
| 0 | 0 |    0|


With our current configuration, the output is completely off, the only way we can tune our perceptron is by start changing the weights, the weights is the only variable that we can change.
We can start changing weights and start trying if the desired output match exactly with the objective output. let's try one time changing the weights:

**new set of weights**

we can see that the result is not our expected output, we can keep trying but this will be too cumbersome and there is a better way to achieve this and that is learning the weights, to be able to do that we need aside from our score function a loss function.

**Loss Function**
You will start seeing this pattern a lot on machine learning models, we have a score function accompanied by a loss function, both are necessary for the learning to happen as we will see. For now, lets concentrate on what a loss function is.

The loss function will tell us how bad or good our **score** function is behaving, as simply as that. There are a lot of loss functions from simply ones to more complex, here we are going to see the simplest loss function in Machine Learning in this case just the difference between our target and predicted input.

**square error equation without squares**

this function is making a subtraction of our predicted output to the target value, the power of two happens just to have a positive difference.

now if we apply our loss function to our current outputs we will have a value this value will tell us if our perceptron is correct or not and by how much.

Now that we have our loss function in place we can start with the weight learning to do that we need a learning procedure.

**The Perceptron Learning Procedure**

We start by making our weighs randomized
We make an evaluation of all our inputs to our current perceptron with our current weight configuration
Our perceptron will output 1 or 0
We will see if this result is correct or not correct by applying our loss function
If our loss function is >0 that means that our predicted output was incorrect, and if the expected output was 1, then we will subtract our input vector to the current weight configuration.
If our loss function is >0 that means that our predicted output was incorrect, and if the expected output was 0, then we will add out input vector to the current weight configuration.

Now we need to do this learning algorithm a couple of times and our weight will be configured correctly.

