---
layout: post
title: An easy introduction to Machine Learning
published: true
---

Maybe you have already heard about machine learning and some of the amazing things it can do, in this series of posts im going to explain in a very easy way what Machine Learning is and why it is so important for the future of technology.

Machine learning covers a lot of topics and can have a lot of ramifications, but I like to define it as **a set of techniques and specialized algoritms that generate mathematical functions from pure data**, the output of these generated functions are of our special interest because they can give us very good estimates about new data based on previous knowledge, in other words this generated function will try to describe the data no matter how is structured.

An easy way to understand machine learning is to think in mathematical functions, for example when you write a program using a 3rd generation language like C# or Java, you think in variables, if-then statements, classes, methods, generics, loops etc. Now **our goal** is to create a program that outputs the computer **price** in USD dollars depending on the following characteristic:

	• Cpu Speed

And you have the following data-set available, in this case the **price** is given to you in terms of CPU Speed, here we say that the CPU Speed is a feature of the computer price because it influence the price:

| Cpu Speed(Ghz)  |  Computer Price (usd)  |
|:----------------|:-----------------------|
| 	    5.2	      | 	      3000         |
|       4.4       |           2700         |
| 	    3.4	      |           2500         |
|     	2.1	      |           600          |

Thinking in code we want to create a function like this:

```
function  int CalculatePrice(int cpuSpeed){
	/*implementation*/
	return price;
}
```

Now our goal is to implement the body of this function, we can have some insights from the data, for example is obvious that a high CPU speed appears to increase the total price of the computer this is they are directly proportional.

So we could write some code based on this insight:

```
function int CalculatePrice(int cpuSpeed){
	
    var cpuSpeedFactor = ? ; /* our guess */
    var price = cpuSpeed * cpuSpeedFactor;
    
    return price;
}
```

What value should the variable _cpuSpeedFactor_ needs to be so that our function outputs the exact same values of our data-set? Ej:

```
//compare it to our data-set
CalculatePrice(5.2); // outputs 3000
CalculatePrice(4.4); // outputs 2700
CalculatePrice(3.4); // outputs 2500
CalculatePrice(2.1); // outputs 600

```
Because this is a very small data set we can try with different values and come up with a acceptable solution, for example lets say that **cpuSpeedFactor = 200**

{% raw %}
  $$5.2*200=1040 \text{ expected 3000}\\ 4.4*200=800 \text{ expected 2700}\\ 3.4*200=680 \text{ expected 2500}\\ 2.1*200=420 \text{ expected 600}$$
{% endraw %}

Lets calculate the total offset of our estimated values vs the values from the data set (the real values)

{% raw %}
  $$\lvert(3000-1040)\rvert+\lvert(2700-800)\rvert+\lvert(2500-680)\rvert+\lvert(600-420)\rvert=6100$$
{% endraw %}

Our function predictions have an offset of 6100 usd dollars testing with our data set.

Now lets try another value for cpuSpeedFactor for example **cpuSpeedFactor = 600**

{% raw %}
  $$5.2*600=3120 \text{ expected 3000}\\ 4.4*600=2640 \text{ expected 2700}\\ 3.4*600=2040 \text{ expected 2500}\\ 2.1*600=1260 \text{ expected 600}$$
{% endraw %}


{% raw %}
  $$\lvert(3000-3120)\rvert+\lvert(2700-2640)\rvert+\lvert(2500-2040)\rvert+\lvert(600-1260)\rvert=1300$$
{% endraw %}

Our function predictions now have an offset of 1300 usd dollars this time, much better! but is there an optimum value for cpuSpeedFactor that produces the minimum offset (this is to be as close as possible to our data-set)?


## Statistics to the rescue 
It turns out that statistics can helps us a lot here. Statistiscs uses mathematical methods that can adjust to data to help gathering adittional insights. Let me explain you the most basic (but powerful) model in statistics, linear regression.

## Linear Regression
Linear Regression is a mathematical model that tries to fit a mathematical formula of a straight line to a set of given points of data in a two-dimensional plane.

For example lets draw our given data-set (cpu speed vs computer price).
(diagram)

In this case you have a bunch of points representing the relationship between cpu-speed and price, now with linear regression you can model an ecuation that fits your data using a straight line, this line has the form of an ecuation:

{% raw %}
  $$y = x*b+c$$
{% endraw %} 

where C is a constant that allows the straight line to translate vertically, x is the independent variable in this case the _cpu_speed_ and y the dependant variable in this case the _price_.

What linear regresion does is to try to create a linear relationship between our two variables based on all data points that exist on the plane, in this case _cpu_speed_ and _price_, linear regression calculates the minimun distance for each data point to a straight line that covers the entire set of data, for example:

![Computer Price vs Cpu Speed]({{site.baseurl}}/_posts/PriceVsCPu.png)

A straight line that try to cover the entire set of data points is really useful because we have an explicit guide that represent our data points and because this line is continuous our function can output not only the values from the data-set but a lot of more values. For example we know that for cpu-speed of 5.2 Ghz the price is 3000 usd dollars but what about a new computer of speed 5.5 Ghz? if we use our straight line as a reference we can get a good estimate, our funtion gives us a new ability to estimate new unseen values from our fixed data, in statistics and machine learning this ability is called prediction.

## The prediction/estimation concept
This same prediction ability gives us the capability to generalize, that is to create a function that represents the entire set of data points and also represent the hidden rules and patterns that our data have implicitly, so that when a new point comes to our function the new output will have coherence. A straight line generalizes very well to all kind of data, but that can be bad in some cases because if our data is very complex our function will only be able to represent it with a line giving us wrong predictions this is a well known machine learning problem called _underfitting_, in this case when the data is very complex our model is not powerfull enough to be able to fit the fixed data set.

For example with our modeled data and our model (rule) fitted to our data we already have an ecuation 

 {% raw %}
  $$y = 100*x$$ + 76
 {% endraw %}

fitted to our data, if we want to know the income of a new age value (a value that we never registered in our data points) we can predict it or what i like to call it better "estimate it", this value is using our statistical linear regression model will give us a very good aproximate value.
I like to think that Linear Regression is the most basic form of Machine Learning, because when you research more on the topic you can somehow see that even the most complex Machine Learning Algorithms at a deep level are trying to  behave the same like Linear Regression, that is, trying to model a mathematical ecuation from data to try to describe in the best way posible the data set.

Of course the value that our model gives us is not exact, it is an aproximate, and in some cases this is very well accepted.



Cpu Speed
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
