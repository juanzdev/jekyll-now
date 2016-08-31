---
layout: post
title: An easy introduction to Machine Learning
---

If you are into technology maybe you have heard about machine learning, in this series of posts im going to explain in the easiest manner what Machine Learning is and why it is so important for the future of technology.

Natural Pattern Recognition
Data is everywhere nowdays, everything with our eyes can somehow be data in a computer, for example your eyes right now are seeing a computer or a smartphone screen, this can be translated to a image in a computer, this is our way to put realworld concepts to computer data.

Our brains can recognize the concept or the idea of what our eyes are seeing, this supernatural ability is being used since the very first moment we born. When a baby is born somehow is not aware of the external world, the baby doesnt know what  a car or a toy is, actually he event doesnt know the very same concept of a toy or a car. But since the very first moment this child is born his eyes begin to capture a lot information from the realworld everytime, every single second, and this information (if we can call it information) flows directly to the brain, and it turns out that somehow because of evolution, the brain is the most sofisticated system to recognize new patterns and get familiarized with them, the learning ability of the brain allows us to become better and better everyday at new unknown tasks.

Let me give you a clear example of this. Taking the same baby example, this baby is in his cradle and got a toy this toy is a car, he doesnt know it is a car until the mother tell him pointing to the object "this is a car", after a repetition process the idea of a car referencing this unknown object sticks in the babys brain, the shape of the toy car, the color, the characteristics, the brain has learned the concept of what a car is.

Thanks to this exposure and the awesome capability of our brains to learn new things and learn new patterns everyday we can easily recognize almost any abstract concept of our reality. 

The Problem of data and computer pattern recognition
For engineers and software developers to create computer programs that emulate the same recognition capabilities of our brains has been a topic of research for over the last 50 years. It is a very challenging task to create something that evolution has perfectionated over the centuries.

When you think in how to create a program that could recognize the shape of a car from an image it is almost imposible to think in if-else statements or  to think in some special techique using a 3rd generation language programming like Java or C#. And it turns out that image recognition before machine learning used this bruteforce methods that are actually pretty clever if you think about them but still feels like a very hard thing to do and not a very natural way to solve the problem.

If you think at a deep level from the computational perspective, the ability to recognize the concept of a car from a image is the ability to recognize the pattern of a car in a set of color pixels aka recognize a pattern in a set of data.

If there were a way to create a program that could recognize image patterns from data in the best way possible what will be this implementation? What programming techniques will use? Should we use loops? Recursion? If then statements? Some fancy technique? 
It turns out that this question is very hard to answer, in fact almost impossible to answer these days,  we dont have a method or technique to create a program that could do such task at least with human performance level.

Statistics to the rescue 
It turns out that the field of statistics is the one to extract insights from the data, and statistics method for data analysis help a lot with this challenging task to aid computers with pattern recognition tasks.

Statistiscs uses the concept of models, models are mathematical methods that can adjust to data to help the statistician or analysit to get insights about the data in a easier way. Let me explain you the most basic model (but powerfull) in statistics, the Linear Regression Model.

Linear Regression
This simple but  powerfull model and still used nowdays can model a mathematical formula to estimate values from a set of data.

For example lets say you gather some data, let say you ask a group of people their income and their age and you take this data and represent them in a cartesian diagram you will have a diagram in 2 dimensionns where you can contrast the age vs income.




In this case you have a bunch of points representing the realtionship between age and income, now with statistics and more specifically with a model like linear regression you can model a custom formula for your data that describes best this relationship, this formula will model a linear relationship that will follow  the  y = c + b*x  form where c is a constant (that gives the ability of the realtionship to move or translate on the y axis) , x is the independent variable say the age and y the dependant variable say the income, in this case we are assuming that the income in someway  is affected by the age of the person.

What linear regresion does is to try to create a linear relationship between our two variables , in this case age and income, to do this and without going too deep on how it do it, linear regression calculates the minimun distance for each data point to a straight line that covers the entire set of data points, for example



Here you can see that the sum of the  vertical distance from the data point to the straight line is the minimun, linear regression gives us the minimun sum of this distance and hence gives us the best possible straight line that pass through the set of data points.
The ecuation that describes this straight line has the form y = c + b*x and it actually tries to describe in the best way possible  with a straight line the relationship between the two variables. If you think of this, it is actually pretty usefull because you can think of trends or a more global picture about your specific data. You somehow have a rule that describes very good your data.

The prediction/estimation concept
This is kind of a magical word, but when you see it in terms of statistics , for example in the case of linear regression it is actually pretty natural and simple word, for example with our modeled data and our model (rule) fitted to our data we already have an ecuation 

y = 76+ 100*x

fitted to our data, if we want to know the income of a new age value ( a value that we never registered in our data points) we can predict it or what i like to call it better "estimate it", this value is using our statistical linear regression model will give us a very good aproximate value.
I like to think that Linear Regression is the most basic form of Machine Learning, because when you research more on the topic you can somehow see that even the most complex Machine Learning Algorithms at a deep level are trying to  behave the same like Linear Regression, that is, trying to model a mathematical ecuation from data to try to describe in the best way posible the data set.

Of course the value that our model gives us is not exact, it is an aproximate, and in some cases this is very well accepted.


What did just happen was Gold
In case you realized this statistical model did a pretty awesome thing, it modeled itself to the data!, if you translate this concept to computer algorithms we just created a program that took an initial data to write itself, yes i know is not that it create custom code or anything but it aproximated a mathematical function from the data, let me repeat this, it aproximated a mathematical function from the data. If you think in computer programs, well they are somehow like mathematical functions, they take an input or a series of inputs and produces and output,  a mathematical function is the same. If we have our estimation tool (in this case our linear regression model) and pass it another set of data  that has nothing to do with income vs age , for example house prices vs m2 , and pass it to our statistical model, it will accomodate this time to this new data! and will generate its own mathematical funcion. What did just happen was that we have a way to create functions from data only, without thinking in the funcion implementation itself!.

This is the key to machine learning or what i like to call it better statistical learning, the key is to aproximate mathematical functions from data, here we humans dont have to think about the function and how to implement it. This is awesome you know because we humans athough have our brains and our mechanism to recognize patterns we cant just translate this knowledge to computer programs using standard programming. But we can create programs that tune themeselves using just data!.


I would like to stop here, you know machine learning can be overwhelming but the main objective of this blog post was to take you to the most basic thing in machine learning but also the most important one.

Please stay tunned for the next series of Machine Learning posts.
