---
layout: post
title: An easy introduction to Machine Learning
published: true
---

Maybe you have already heard about machine learning and some of the amazing things it can do, in this series of posts im going to explain in a very easy way what Machine Learning is and why it is so important for the future of technology.

Machine learning covers a lot of topics and can have a lot of ramifications, but I like to define it as a set of techniques and specialized algoritms that generate mathematical functions from pure data, the output of these generated functions are of our special interest because they can give us very good estimates about new data based on previous knowledge, in other words this generated function will try to describe the data no matter how the data is structured.

An easy way to understand machine learning is to think in mathematical functions, for example when you write a program using a 3rd generation language like C#, you think in variables, if-then statements, classes, methods, OOP etc. Lets say for example that you want to create a program that outputs the computer **price** in USD dollars depending on the following characteristics:

	• Cpu Speed
	• Memory Ram
	• Cache
	• Flops
	• Reads per second
	• Writes per second

You have the following data available for you to write your program in this case the **price** is also given to you:

| Cpu Speed(Ghz)  | Ram (Gb)  | Cache (Mb)  | Flops  |  Reads Per Second |  Writes per Second | Computer Price (usd)  |
|:----------------|:----------|:------------|:-------|:------------------|:-------------------|:----------------------|
| 	9  	  | 	10    |      2      |  3500  |       4500        |        1500        |        3000           |
|       7         |     7     |      1      |  2500  |       2500        |        1500        |        1500           |
| 	6 	  |     7     |      2      |  1500  |       4500        |        1500        |        2500           |
| 	5	  |     2     |      1      |  1000  |       3000        |        1500        |        1000           |

Specifically we want to create a function like this

```
function  int CalculatePrice(int cpuSpeed, int ram, int cache, int flops, int reads, int writes){
	/*implementation*/
	return price;
}
```

Now our goal is to implement this function, we can have some insights from the data, for example you can see that a high CPU speed appears to increase the total price of the computer, the memory ram also seems to be an important factor for the price.

So we could write some code based on this insight:

```
function int CalculatePrice(int cpuSpeed,int ram,int cache, int flops, int reads, int writes){
	
    var cpuSpeedFactor = 9;
    var ramFactor = 8;
    var cacheFactor = 6;
    var flopsFactor = 5;
    var readsFactor = 4;
    var writesFactor = 3;
    
    var price = cpuSpeed*cpuSpeedFactor+ram*ramFactor+cache*cacheFactor+flops*flopsFactor+reads*readsFactor+writes*writesFactor;
    
    return price;
}
```
I ve created some _factor_ variables that gives each corresponding parameter a _weight_  the bigger the factor the more it influences the total price.
Also, I ve have assigned a  higher value to the cpuSpeedFactor variable based on my own intuition, this function will give us an aproximation based on my own judgment. The problem with this approach is that we dont know how to specify the constant values or weights for each variable we are just guessing here, yes I know that based on the data we have that CPU speed have a high impact on the computer price, but we dont know if this impact have a value over the total price of a factor of 9. The same applies for the rest of the factor variables, we dont have any real clue how they influence the final computer price.

Maybe we can come up with a conbination of factor variables in such a way that they represent our data exactly. But, ¿what if we have to guess the price of a new computer with new specs that are not registered on our data-set?, ¿what price will our custom function output?, ¿it will be a precise value?, ¿will our ouput have coherence with the dataset?.
As you can see its very difficult to extract logic from data, maybe for this small data set of 4 rows is achievable, but ¿what if we have a data set of +100 rows? it would be extremelly difficult to think in some function that represents the dataset correctly.

## Statistics to the rescue 
It turns out that statistics can helps us a lot here. Statistiscs uses mathematical methods that can adjust to data to help gathering adittional insights. Let me explain you the most basic (but powerful) model in statistics, linear regression.

## Linear Regression
Linear Regression is a mathematical model that tries to fit a mathematical formula of a straight line to a set of  given points of data in a two-dimensional plane.

For example lets say you gather some data, let say you ask a group of people their income and their age and you take this data and represent them in a cartesian diagram you will have plot of points in 2 dimensions where you can compare age vs income values.

In this case you have a bunch of points representing the relationship between age and income, now with statistics and more specifically with a model like linear regression you can model a custom formula for your data that describes best this relationship, in the case of linear regression it will be a straight line, this formula will model a linear relationship that will follow  the form:

{% raw %}
  $$y = $$b\*$$x+C
{% endraw %} 

where C is a constant that allows the straight line to translate vertically, x is the independent variable say the _age_ and y the dependant variable say the _income_, in this case we are assuming that the income in some way is affected by the age of the person.

What linear regresion does is to try to create a linear relationship between our two variables based on all data points that exist on the plane, in this case _age_ and _income_, linear regression calculates the minimun distance for each data point to a straight line that covers the entire set of data, for example:

(diagram)

A straight line that try to cover the entire set of data points is useful because we have a formula or a guide to represent data points that were not previusly seen. For example we can know the income of the age 34 and get a fair result, note that the age 34 was never contemplated in our set of data points, this is what is called in statistics and machine learning predict a value.

## The prediction/estimation concept
This same prediction ability gives us the capability to generalize, that is to create a function that represents the entire set of data points and also represent the hidden pattern that is happing behind the scenes so that when a new point comes to our function the output will have coherence. A straight line generalizes very well to all kind of data, but that can be bad in some cases because if our data is very complex our function will only be able to represent it with a line giving us wrong predictions. Sometimes the data is very complex and a linear regression is not enought to be able to  predict new values.

For example with our modeled data and our model (rule) fitted to our data we already have an ecuation 

 {% raw %}
  $$y = 100*x$$ + 76
 {% endraw %}

fitted to our data, if we want to know the income of a new age value (a value that we never registered in our data points) we can predict it or what i like to call it better "estimate it", this value is using our statistical linear regression model will give us a very good aproximate value.
I like to think that Linear Regression is the most basic form of Machine Learning, because when you research more on the topic you can somehow see that even the most complex Machine Learning Algorithms at a deep level are trying to  behave the same like Linear Regression, that is, trying to model a mathematical ecuation from data to try to describe in the best way posible the data set.

Of course the value that our model gives us is not exact, it is an aproximate, and in some cases this is very well accepted.
