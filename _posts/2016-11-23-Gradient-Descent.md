---
published: false
---
---
layout: post
published: true
title: Gradient Descent
---

## What is Gradient Descent
Gradient descent is a very usefull algorithm in machine learning and mathematics because it allows your calculate the local minima or maxima of a function, lets remember that a function can be represented on a plot as a plane or hyperplane depending of the number of variables it has, this plane can be imagined as a nature horizon with mountains going up and holes going down, each point on the plane is the result of the function evaluated on some specific values for the  independent variables. The goal here is to look at the highest point on a plane or the lower,  this problem is called finding the global minima or global maxima, gradient descent can help us finding not the global minima or optima but the local minima or optima, it has been proven that finding a local minima or optima is enough for tackle real world problems on machine learning.

Today exist a numerous number of algorithms to find local minima of a function, gradient descent is just one method.

Gradient descent works by deriving the cost function in terms of the parameters, when you derive a funtion in term of a variable the resulting function is exactly the slope over the actual function, with this slope we can guess wether the plane or hyperplane is going up or down depending on the sign of the resulting derived function, having this information we can then know that we must make a small step towards the decreasing slope if what we want is to minimize the function, also the size of the step can be configured to make small or bigger leaps toward the plane.

Lets remember that gradient descent allow us to find the minima of a function based on certain parameters but only on small steps, there is no way to reach the minimun solution without making some iterations first, actually this is not a problem because you actually practice gradient descent on your training phase just to find the best configuration of your paramters for your current machine learning problem, this happen at training time, once you find the optimum solution you can just plug it onto your predit funcion and you are done.
