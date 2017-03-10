---
layout: post
title: Teeth classifier using convolutional neural networks
published: false
---

In this blog post, I'm going to explain how you can create a complete machine learning pipeline that solves the problem of telling weather or not a person is showing is teeth, I will show the main challenges that I faced implementing this task. Specifically I'm going to use a convination of OpenCV computer-vision models for face detection with a convolutional neural network for teeth recognition.

Main challenges:
1. Finding datasets where people are showing their teeth or not and tailoring them to the problem
    2. Label the data accordingly ( 1 for showing teeth, 0 not showing teeth)
    3. Detect the face region in an image
    4. Detect the principal landmarks on the face
    5. Transform the face with the detected landmarks to have a "frontal face"
    6. Highlight the teeth on the face
7. Augment data for posterior training
8. Seting up the Convolutional neural network in Caffe
9. Training and debugging the system
10.Testing the model

#Finding a dataset
There are a lot of datasets with faces on the web, I choose an open dataset called MUCT database http://www.milbo.org/muct/ , this dataset contains 3755 faces with landmarks, for the purpose of this post I'm not going to use the landmark data.

//picture of the girl picture (3 pictures)

This dataset has a lot of variation in lighting, people face types and expressions. I only used one part of this dataset called muct-a-jpg-v1.tar.gz, this file contains 700+ faces, although this is a small number for training the machine learning model, it is possible to obtain good results using data augmentation, the reason I choose only this limited subset of data is because at some point in the process is necessary to do manual labeling of the data but you can label more and more data to obtain better results.

#Labeling the data

Now there is some manual process involved here but is necessary to do it only once, we are going to label each of the 700 faces with the label (1 for showing teeth, 0 not showing teeth), the label will be stored on the filename of the image. Because this can be a tedious process I created a simple tool for labeling images, if you push the button yes it will add to the existing filename the label _true or _false otherwise, if you want to use this tool for your purposes feel free to pull it from git hub and modify it to your needs here 

//link to GitHub 

//picture of the manual labeling tool in action

Note that I could spend much time labeling the rest of the data (up to 3755 faces) and also search on the web more data but for time constraints I did only this process for the 700 first images, but the bigger the data set the better.

Also note that this labeled data is not our training set yet, because we have such small data set we have to help the model little bit, in this case, we are going to get rid of unnecessary data and noise in the images, that means that we are going to use other already trained models to detect the face region and discard the rest of the image, this will help us to reduce overfiting the data quite a bit.

#Detect the face region

There are different models for face detection, the most well known are Haar cascade and HOG, OpenCV offers a nice and fast implementation of Haar cascades and Dlib offers a more precise face detection algorithm with HOG. After doing some testing with both libraries I found that DLib face-detection is much more precise, the Haar approach gives me a lot of false positives, the problem with Dlib face-detection is that it is slow, I think that you can speed-up the process a little by doing some compilation configuration on the library. Because I wanted to test my Teeth detector on real video HOG face detetion was too slow to be practical so I ended up using OpenCV for this task. I'm going to use a method called Histogram of Gradients or HOG to transform the image to another representation of values for easy interpretation then I will extract the landmark features of the face and finally I will transform the face using the landmark to have a frontal face, lucky for us those three steps can be simplified a lot by using the dlib library.
Note that you can also use a convolutional neural network for face detection, in fact you will get much better results if you do, but for simplicity I will stick with OpenCV for simplicity.

In Python I will create two clases, one for OpenCV face detection and one for DLib face detection. These clases will receive an input image and will return the area of the face.

//code fragment for OpenCV

//code fragment for HOG

Now a tipicall problem in computer vision is the different transformations our images can have, they can be rotated at certain degree, and that will make dificult for our machine learning model to predict the correct answer, although it is proven that in a Convolutional neural network you can have certain degree of flexibility with rotation and translations, this is a nice property of CNNs called invariance to transformations. We are going to have that nice property only for our teeth detector ConvNet but for face detection we have to make an addional process called landmark detection to overcome this problem.

//code fragment for landmark detector

//image of landmarks with sample face

this function receives the face region and will detect 68 landmark points using a previuslly trained model this will help us have relevant information about the orientation of the face.
Now we can make a warp transformation to the face using the landmarks as guide to have the face image facing front.

//code fragment for affine transformation
//image of face affined

Now that we have a standard way to see images our face detection process will be much precise.
The next step is to slice the image horizontally and take only the botton region this is the mouth region.

Now that we have frontal faces we can focus on the mouth region, a simple region is to divide this image in two and take the bottom region, for sake of simplicity I'm going to use this approach.

//code fragment for image slicing
//image of mouths parts
//picture of a bunch of mouths

#Highlithing the teeth 
Now a quick technique to highlight the teeth on the mouth region is inverting the image pixels, this is converting the image to the negative image, there may be other methods but this particulary one worked well for me.

//code for extracting the negative image
//picture of mouth enhanced with negative pattern
//picture of a bunch of negative mouths

This pre-processing step will help a lot our convolutional neural network to learn features easier, but also note that we are doing this because in this particular case we are working with small sets of data, if we had millions of images we could easily feed the entire image to the net and it will learn teeth features for sure.

#Data Augmentation
Because we are working with small sets of data we need to create syntetic data aka data augmentation, for this particular case we are going to do the following transformations for each image in our dataset to get almost 10x times more data.

* Mirroring of mouths
Foreach image we are going to create a mirrored clone, this will give us 2x the data.
//example of mirroring with a muct image
* Shearing of mouths
Foreach image we are going to make small rotations, specifically -30,-20,-10,+10,+20,+30 degrees of rotation this will give us 8x the data.
//example of shearing with a muct image
* Scaling of mouths
Foreach image we are going scale it by small factors, this will give us 2x the data
//example of scaling with a muct image

//code for mirroring

//code for shearing

//code for scaling

#Seting up the Convolutional neural network in Caffe

#Preparing the training set and validation set
Now that we have our data ready, we need to split it into two subsets, we are going to use an 80/20 rule, 80 percent of our transformed data is going to be the training set and the rest of the 20 percent is going to be the  validation set. The training data will be used during the training phase for learning and the validation set will be used to test the performance of the net during training, in this case, I move the data accordingly to their respective folders training_data and validation_data

//picture of data in corresponding folder

#Creating the LMDB file

With the data in place, we are going to generate two text files, each containing the path of the image plus the label (1 or 0), these text files are needed because Caffe has a tool to generate LMDB files for you just with these plain text files.

//code for generating those text files

//image of the plain text showing the format

Now that we have the two text files, we are ready to generate the LMDB file, the LMDB file is a database file that stores all our training data along with their respective labels.

To generate both training and validation LMDB files do the following:

convert_imageset --gray --shuffle /devuser/Teeth/img/training_data/ training_data.txt train_lmdb
convert_imageset --gray --shuffle /devuser/Teeth/img/validation_data/ training_val_data.txt val_lmdb

#Extracting the mean data for the entire dataset
A common step in computer vision and image processing is to extract the mean data of the entire training data to ease the training process, to do this:

compute_image_mean -backend=lmdb train_lmdb mean.binaryproto

This will generate a file called mean.binaryproto, this file will have matrix data related to the overall mean of our training set, this will be substracted during training to each and everyone of our training examples

#Designing and implementing the Convolutional Neural Net
For quick prototyping, I'm going to use the Caffe Deeplearning framework, but you can use other cool frameworks like TensorFlow or Keras.

Convnets are really good at image processing because they can learn features automatically just by providing input and output data, they are also very good at transformation invariances this is to small changes in rotation and full changes in translation.
In Machine Learning there are a set of well-known architectures for image processing like AlexNet, VGGNet, Inception etc. If you follow that kind of architectures is almost guaranteed you will obtain the best results possible, for this case and for the sake of simplicity and training time I'm going to use a simplified version of AlexNet with much less convolutional layers, remember that here we are just trying to extract Teeth features from the face and not entire concepts of the world like AlexNet does, so a net with much less capacity will do fine.

//code of 3 prototxt
//image of architecture

To train the neural network
caffe train --solver=model/solver_feature_scaled.prototxt 2>&1 | tee logteeth_ult_fe_2.log


//image of loss vs iterations with 10000 iterations

#Testing the trained model with unseen data

//false positive false negative chart

Accuracy precision and recall
//code
F1score
//code

Now we have metrics to benchmark our trained model, with this in place we can quickly start tweaking things in our model or experimenting with different approaches and at the end see the final improvement with a number.


predict_feature_scaled.py


Testing the image by moving them to the coorect folder

Testing an individual image 


Testing our net with real video!
although training a convnet is a very slow process, testing it is not!, in fact, it takes milliseconds to test the trained model, to prove you that I'm going to call the trained net in each frame of a video to show the predictions on realtime.

//gif of a video running with the net showing if the person is showing the teeth or not
