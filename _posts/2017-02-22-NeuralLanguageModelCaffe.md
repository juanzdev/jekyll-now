---
layout: post
title: Neural Language Model with Caffe
published: true
---

In this blog post I will explain how you can implement a Word Embeddings implementation in Caffe using Bengio Neural Model and based on Hinton Coursera course code in Octave. This is just a practical fun exercise I made to see if it was possible to model this problem in the Caffe deep learning framework.

The problem that this model is trying to solve is creating a Neural Model that is capable to predict the next word given three previous words, the predicted word needs to make sense according to the previous context, and with training data the model will be able to learn knowledge about word relationships and at the end it will be very capable of doing this task.

A neural network is compossed of what is called hidden layers, the state of this layers will be unknown, they are very useful because they are a representation of our data in different dimensions, this is the same as saying that the hidden layers are transformation of our features, this is commonly called different feature representations.

By working with different feature representations is possible to create very abstract associations of the data, for example when training a classifier of pictures we can create boundaries in high dimensional space to group "concepts" of pictures and be able to differentiate how similar or diferent a picture is to other picture, this is possible because in those high dimensions the concept of a picture can be represented with a bunch of number combinations in a high dimensional data space and in this high dimensional space we can start talking about distances between high dimensional vectors to see how close a vector relates to each other.

## Word Embeddings
A high dimensional representation of a word is called a word embed, by having a different feature representation of the words we can group words by similarity not in syntax but in context.

Bengio was the first to propose an neural architecture for the word embedding problem:

-image of bengio neural net architecture-

This net works by the premisse that you have a static vocabulary with word codes (250 codes in total), this is, for each word we are going to assign it a unique code, in this case an integer. Then a look-up operation will take place over the embeed matrix, this means that initially this matrix will have random values but after the back-propagation learning procedure each row of this matrix will represent the different and expanded feature representation of each word of the vocabulary, this is the reason why this matrix is 250 by 50, in this case the number 50 is the expanded representation of each word, this is an hyperparameter and you can try with different values.

The net is followed then by a hidden layer with 200 neurons in a fully connected fashion, this layer will help to add more complexity to our internal representations by being more flexible with more non-linearities.

The net is then follwed by a softmax layer to be able to represent the final result with probabilities, at the end of the this net we are going to have the most probable words that come next to our previous three.

## Caffe Implementation
The main goal of this exercise was to be able to create this neural net to Caffe based on the code already provided by Hinton in his Neural Network course.

1. HDF5 Data Extraction 
The first thing we have to do is bulk all our training, validation and test data to a HDF5 file, this is one of the files that Caffe supports for data, HDF5 format is recommended on Caffe when we are not using image data.
Caffe is mainly a deep learning framework oriented towards image processing but they state that is perfectly fine to use non image data. 

Because the initial data is on a .mat format on octave is necessary to export this to CSV, this is Octave code:

```python
%generate dataset from octave
[train_x, train_t, valid_x, valid_t, test_x, test_t, vocab] = load_data(372500);
csvwrite("train_x.csv",train_x')
csvwrite("train_t.csv",train_t')
csvwrite("valid_x.csv",valid_x')
csvwrite("valid_t.csv",valid_t')
csvwrite("test_x.csv",test_x')
csvwrite("test_t.csv",test_t')
```

This command exports our data file to a nice format:

--picture of comma separated values

Now towards the code, the first script we have to do is read all this csv data and store it in a HDF5 compatible format

```python
import h5py, os
import numpy as np
import csv

trainXFilePath = 'csv/train_x.csv'
trainTFilePath = 'csv/train_t.csv'
testXFilePath = 'csv/test_x.csv'
testTFilePath = 'csv/test_t.csv'

#read csv training data
with open(trainXFilePath,'rb') as f:
	reader = csv.reader(f)
	data_as_list = list(reader)

#shape (372550, 3)
data = np.array(data_as_list).astype("int")

#read csv training data labels
with open(trainTFilePath,'rb') as f:
	reader = csv.reader(f)
	data_as_list = list(reader)

#shape (372550, 1)
labels = np.array(data_as_list).astype("int")

#create HDF5 file with training data and labels
f = h5py.File('hdf5/train.h5', 'w')
f.create_dataset('data', data = data)
f.create_dataset('label',  data = labels)
f.close()

#read csv test data
with open(testXFilePath,'rb') as f:
	reader = csv.reader(f)
	data_as_list = list(reader)

data = np.array(data_as_list).astype("int")

#read csv test data labels
with open(testTFilePath,'rb') as f:
	reader = csv.reader(f)
	data_as_list = list(reader)

labels = np.array(data_as_list).astype("int")

#create HDF5 file with test data and labels
f = h5py.File('hdf5/test.h5', 'w')
f.create_dataset('data', data = data)
f.create_dataset('label',  data = labels)
f.close()
```

If you follow into the code you can see that this simple script only save all our training data into an object called data and all our label data into an object called label, this is required by Caffe to know where to read data and labels.

--picture of .h5 files--

As you can see this files are all binary, now the next step is to start creating our neural network architecture in Caffe:

2.  Caffe Neural Network
Creating a net in Caffe requires to write all the definition in prototxt files, this files have a JSON notation so is very easy to write them.

The first prototxt file we need to create will be called train_val.prototxt here we will store all the architecture for our netural network this includes all the layers and what types of neurons will have, also here we are going to define the data layer that will read our HDF5 files. This file is called train_val because here will be defined the architecture for the training phase as well as for the validation phase.

```json
name: "LanguageNet"

layer {
	name: "4gramsDataSet"
	type: "HDF5Data"
	top: "data"
	top: "label"
	hdf5_data_param {
		source: "model/train.txt"
		batch_size: 100
		shuffle: true
	}
	include: { phase: TRAIN }
}

layer {
	name: "4gramsDataSet"
	type: "HDF5Data"
	top: "data"
	top: "label"
	hdf5_data_param {
		source: "model/test.txt"
		batch_size: 100
		shuffle: false
	}
	include: { phase: TEST }
}

layer{
	name: "ipWordEmbedding"
	type: "Embed"
	bottom: "data"
	top: "ipWordEmbedding"
	embed_param {
		input_dim: 251
		num_output: 50
		weight_filler {
    		type: "xavier"
    	}
    	bias_filler {
      		type: "constant"
      		value: 0
    	}
	}
}

layer{
	name: "ipHidden"
	type: "InnerProduct"
	bottom: "ipWordEmbedding"
	top: "ipHidden"
	inner_product_param {
		num_output: 200
		weight_filler {
     		type: "xavier"
    	}
    	bias_filler {
      		type: "constant"
      		value: 0
    	}
	}
}

layer {
  name: "relu1"
  type: "Sigmoid"
  bottom: "ipHidden"
  top: "reluOutput"
}

layer{
	name: "inputToSoftmax"
	type: "InnerProduct"
	bottom: "reluOutput"
	top: "inputToSoftmax"
	inner_product_param {
		num_output: 251
		weight_filler {
     		type: "xavier"
    	}
    	bias_filler {
      		type: "constant"
      		value: 0
    	}
	}
}

layer{
	name: "SoftmaxLoss"
	type: "SoftmaxWithLoss"
	bottom: "inputToSoftmax"
	bottom: "label"
	top: "loss"
}

layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "inputToSoftmax"
  bottom: "label"
  top: "accuracy"
  include: { phase: TEST }
}
```

The code is self-explanatory althouth there are some important things to highlight:

the first layer of type HDF5Data reads a train.txt file and not our HDF5 file directly, this txt file has the path of the HDF5 file, that just how Caffe works.
You can see that we have two HDF5Data layers, this is because for the train phase we are going to use a different dataset than the test phase, the test phase will reference the HDF5Data that corresponds to our test set.

Another thing to highlight is the Embeed layer, althouth the documentation is very scarce about this type of layer on the web I think I managed to make it work correctly, this layer will specify the input dimensions to 250 this will be our total vocabulary and the expanded vector of 50 as specified by the output property, this means that this layer will store on a blob a matrix of 250 by 50 where each row or word will have a different feature representation of 50 dimensions, this layer will do a lookup operation for each index in our vocabulary, this functionallity is required to be able to implement a word-embedding functionallity correctly.
