---
layout: post
title: An easy introduction to Machine Learning
published: true
---

Maybe you have already heard about machine learning and some of the amazing things it can do, in this series of posts im going to explain in a very easy way what Machine Learning is and why it is so important for the future of technology.

Machine learning covers a lot of topics and can have a lot of ramifications, but i like to define it as a set of techniques and specialized algoritms that generate mathematical functions from pure data, the output of these generated functions are of our special interest because they can give us very good estimates about new data based on previous knoweledge, in other words this generated function will try to describe the data no matter how the data is structured.

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

So we could write some aproximation based on this insight:

```
function  int CalculatePrice(int cpuSpeed,int ram,int cache, int flops, int reads, int writes){
	
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

We assigned some random but higher value for the cpuSpeed variable, this function will give us  an aproximation based on our judgment and on our own insights. The problem with this aproach is that we dont know how to specify the constant values or weights for each variable, yes we know that based on the data we know that cpu speed have a high impact on the computer price, but we dont know if have an impact of a factor o 9 on the total function result. Now we are just taking into account just the cpu speed variable but what if the rest of the variables values have influence in the computer price?. This simply task is getting harder and harder.
What if we have a new computer with new specs that are not registered on our data-set, what price will our custom function will output?, i will be a precise value?
As you can see its very difficult to extract logic from data, maybe for this small data set of 4 rows is achievable but what if we have a data set of +100 rows it would be extremelly difficult to think in some function that represents the dataset correctly.

## Statistics to the rescue 
It turns out that the field of statistics is the one to extract insights from the data.
Statistiscs uses the concept of models, models are mathematical methods that can adjust to data to help the statistician to get insights about the data in a easier way. Let me explain you the most basic model (but powerfull) in statistics, the Linear Regression Model.

## Linear Regression
This simple but  powerfull model and still used nowdays can model a mathematical formula to estimate values from a set of data.

For example lets say you gather some data, let say you ask a group of people their income and their age and you take this data and represent them in a cartesian diagram you will have a diagram in 2 dimensions where you can contrast the age vs income.



In this case you have a bunch of points representing the relationship between age and income, now with statistics and more specifically with a model like linear regression you can model a custom formula for your data that describes best this relationship, this formula will model a linear relationship that will follow  the  y = c + b*x  form where c is a constant (that gives the ability of the realtionship to move or translate on the y axis) , x is the independent variable say the age and y the dependant variable say the income, in this case we are assuming that the income in someway  is affected by the age of the person.

What linear regresion does is to try to create a linear relationship between our two variables , in this case age and income, to do this and without going too deep on how it do it, linear regression calculates the minimun distance for each data point to a straight line that covers the entire set of data points, for example


Here you can see that the sum of the  vertical distance from the data point to the straight line is the minimun, linear regression gives us the minimun sum of this distance and hence gives us the best possible straight line that pass through the set of data points.
The ecuation that describes this straight line has the form y = c + b*x and it actually tries to describe in the best way possible  with a straight line the relationship between the two variables. If you think of this, it is actually pretty usefull because you can think of trends or a more global picture about your specific data. You somehow have a rule that describes very good your data.

As you can see linear regression gives us a function that represents a straight line, this has the advantage of generalize the data, this is usefull if we want to predict values because we expect that this function represent the trend of the data. A disadvantage of this straight line is that is limited to represent very basic data, for example if we want to describe a curve relationship it wouldnt be possible using linear regression, but there are a lot of models in statistics that can help us modelate more complex information.

## The prediction/estimation concept
This is kind of a magical word, but when you see it in terms of statistics, for example in the case of linear regression it is actually pretty natural and simple word, for example with our modeled data and our model (rule) fitted to our data we already have an ecuation 

 {% raw %}
  $$y = 100*x$$ + 76
 {% endraw %}

fitted to our data, if we want to know the income of a new age value (a value that we never registered in our data points) we can predict it or what i like to call it better "estimate it", this value is using our statistical linear regression model will give us a very good aproximate value.
I like to think that Linear Regression is the most basic form of Machine Learning, because when you research more on the topic you can somehow see that even the most complex Machine Learning Algorithms at a deep level are trying to  behave the same like Linear Regression, that is, trying to model a mathematical ecuation from data to try to describe in the best way posible the data set.

Of course the value that our model gives us is not exact, it is an aproximate, and in some cases this is very well accepted.
