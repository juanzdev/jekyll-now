---
published: false
---
In this blog post I'm going to explain how you can create a complete machine learning system for teeth detection over a face, I will show the main challenges that I faced implementing this task.

Main challenges:
1. Finding datasets where people are showing their teeth or not
2. Label the data accordingly ( 1 for showing teeth, 0 not showing teeth)
3. Detect the face region in an image
4. Detect the principal landmarks on the face
5. Transform the face with the detected landmarks to have a "frontal face"
6. Highlight the teeth on the face
7. Augment data for posterior training
8. Setup a good architecture for a Convolutional Neural Network
9. Training the data and debugging the system
10. Test the final model

#Finding a dataset
There are a lof of datasets with faces on the web, I choose an open dataset called MUCT database http://www.milbo.org/muct/ , this dataset contains 3755 faces with landmarks, for the purpouse of this post I'm not going to use the landmark data.

//picture of the girl picture (3 pictures)

This dataset has a lot of variation in lighing, people face types and expressions. For the purpouse of this post I only used one part of this dataset called muct-a-jpg-v1.tar.gz, this file contains 700+ faces. Although this is a small number for training a ML model, it is possible to obtain good results using data augmentation, the reason I choose only this subset of data is because at some point in the process is necesssary to do manual labeling of the data.

#Gathering the right data for the training set
Now there is some manual process involved here but is necessary only for the model training, we are going to label each of the 700 faces with the label (1 for showing teeth, 0 not showing teeth), the label will be stored on the filename of the image. Because this can be a tedious process I created a simple tool for labeling images if you push the button yes it will add to the existing filename the label _true or _false otherwise, if you want to use this tool for your purpouses feel free to pull it from git hub and modify it to your needs here //link to github 

//picture of the manual labelling tool in action

Note that I could spend much time labelling the rest of the data (up to 3755 faces) and also start manual labeling more and more faces (you got the idea) but for time constrains I did only this process for the 700 first images.

Also note that this labeled data is not our training set yet, because we have such small data set we have to help the model litle bit, in this case we are going to get rid of unnecessary data and noise, that means that we are going to detect the face region and discard the rest of the image.

#Detect the face region

There are a number of very good methods to substract the face region from an image, in this case I'm going to use a method called Histogram of Gradients or HOG to transform the image to another representation of values for easy interpretation then I will extract the landmark features of the face and finally I will transform the face using the landmark to have a frontal face, lucky for us those three steps can be simplified a lot by using the dlib library.

//code fragment for HOG
//code fragment for landmark detection
//code fragment for face afine

Now that we have frontal faces we can focus on the mouth region, a simple region is to divide this image in two and take the bottom region, for sake of simplicity I'm going to use this approach.

//picture of muct sample facing side
//picture of HOG face
//picture of landmark detection for face
//picture of face afined (facing frontal)
//picture of mouth region 

//picture of a bunch of mouths

#Highlithing the teeth 
Now a quick tecnique to highlight the teeth on the mouth region is inverting the image pixels, this is converting the image to the negative image

//code for extracting the negative image

//picture of mouth enhanced with negative pattern

This pre-processing will help a lot our convolutional neural network, but also note that we are doing this because in this particular case we are working with small sets of data, we had millions of images we could easily pass the entire image to the net and it will surely learn teeth features.







