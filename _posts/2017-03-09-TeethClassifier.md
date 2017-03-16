---
layout: post
title: Teeth classifier using convolutional neural networks
published: true
---

In this blog post, you will learn how to create a complete machine learning pipeline that solves the problem of telling whether or not a person in a picture is showing the teeth, we will see the main challenges that this problem imposes and will tackle some common machine learning problems.
By using a combination of OpenCV computer vision libraries for face detection along with our own convolutional neural network for teeth recognition we will create a very capable system that could handle unseen data withouth loosing significative performance. For quick prototyping we are going to use the Caffe Deep learning framework, but you can use other cool frameworks like TensorFlow or Keras.

The overall steps that will involve creating the detector pipeline are:
1. Finding the correct datasets, and adpating those datasets to the problem
    2. Labeling the data accordingly ( 1 for showing teeth, 0 not showing teeth)
    3. Detecting the face region in an image
    4. Detecting the principal landmarks on the face
    5. Transforming the face with the detected landmarks to have a "frontal face" transformation
    6. Slicing the relevant parts of the image
    6. Easing the data for training
7. Augmenting the data
8. Setting up the convolutional neural network in Caffe
9. Training and debugging the overall system
10. Testing the model with unseen data

# Finding a dataset
## Muct database
We are going to choose an open dataset called MUCT database http://www.milbo.org/muct/, this dataset contains 3755 faces total all unlabeled, all the images were taken on the same studio with same background but with different lighting also the people on the dataset have different expressions so we have some good variety here.

 ![pic](../images/i000qa-fn.jpg)

Because of manual labeling constrains only a subset of this dataset called muct-a-jpg-v1.tar.gz will be used, this file contains 700+ faces and although this is a small number for training the machine learning model, it is possible to obtain good results using data augmentation combined with a powerfull convolutional network, the reason of choosing this limited subset of data is because at some point in the process is necessary to do manual labeling of each picture, but it is always encouraged to label more data to obtain better results.

## LFW database
To have more variety on the data we are going to use the Labeled Faces in the Wild database too http://vis-www.cs.umass.edu/lfw/, this dataset contains 13.000 images of faces total all unlabeled, this database has a lot more variety because it contains faces of people from the web. As same as before we are not going to use the entire dataset but for this case only 1500 faces.

//picture of face of LFW database


# Labeling the data

Labeling the data is a manual and cumbersome process but necessary, in this problem we have to label images from the two face databases, for this particular case we need to label all the faces with the value: 1 if the face is showing the teeth or 0 otherwise, the label will be stored on the filename of the image for practical pourpuses. 

To recap:
For the MUCT database we are going to label 700 faces.
For the LFW database we are going to label 1500 faces.

Manual labeling can be a tedious process so you can use this simple tool for labeling images quickly using hotkeys, if you push the Y key on your keyboard it will add to the existing filename the label _showingteeth, if you want to use this tool for your purposes feel free to pull it from git hub and modify it to suite your needs. 

https://github.com/juanzdev/ImageBinaryLabellingTool

 ![pic](../images/labeltool.jpg)

Note:
Note that this labeled data is not our training set yet, because we have such small data set we need to get rid of unnecessary noise in the images by detecting the face region using some face detection techniques.

# Detecting the face region

## Face detection
There are different techniques for doing face detection, the most well known and accesibles are Haar Cascades and Histogram of Gradients (HOG), OpenCV offers a nice and fast implementation of Haar Cascades and Dlib offers a more precise but slower face detection algorithm with HOG. After doing some testing with both libraries I found that DLib face detection is much more precise and accurate, the Haar approach gives me a lot of false positives, the problem with Dlib face-detection is that it is slow and using it in real video data can be a pain. At the end of the exercise we ended up using both for different kind of situations.

//face detection in action

Note:
You can also use a convolutional neural network for face detection, in fact, you will get much better results if you do, but for simplicity we are going to stick with these out of the box libraries.

In Python, we are going to create two files, one for OpenCV face detection and one for DLib face detection. These files will receive an input image and will return the area of the face.

OpenCV implementation
```python
    def mouth_detect_single(self,image,isPath):

        if isPath == True:
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED) 
        else:
            img = image
        
        img = histogram_equalization(img)
        gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        faces = self.face_cascade.detectMultiScale(gray_img1, 1.3, 5)

        for (x,y,w,h) in faces:
            roi_gray = gray_img1[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if(len(eyes)>0):
                p = x 
                q = y
                r = w
                s = h
                face_region = gray_img1[q:q+s, p:p+r]
                face_region_rect = dlib.rectangle(long(q),long(p),long(q+s),long(p+r))
                rectan = dlib.rectangle(long(x),long(y),long(x+w),long(y+h))
                shape = self.md_face(img,rectan)
                p2d = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)], np.float32)
                rawfront, symfront = self.fronter.frontalization(img,face_region_rect,p2d)
                face_hog_mouth = symfront[165:220, 130:190]
                gray_img = cv2.cvtColor(face_hog_mouth, cv2.COLOR_BGR2GRAY) 
                crop_img_resized = cv2.resize(gray_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
                #cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_rezized.jpg",gray_img)
                return crop_img_resized,rectan.left(),rectan.top(),rectan.right(),rectan.bottom()
        else:
            return None,-1,-1,-1,-1
```

DLIB Implementation using histogram of gradients
```python
    def mouth_detect_single(self,image,isPath):

        if isPath == True:
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED) 
        else:
            img = image
    
        img = histogram_equalization(img)
        facedets = self.face_det(img,1) #Histogram of gradients
        if len(facedets) > 0:
            facedet_obj= facedets[0]
            shape = self.md_face(img,facedet_obj)
            p2d = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)], np.float32)
            rawfront, symfront = self.fronter.frontalization(img,facedet_obj,p2d)
            face_hog_mouth = symfront[165:220, 130:190] #get half-bottom part
            if(face_hog_mouth is not None):
                gray_img = cv2.cvtColor(face_hog_mouth, cv2.COLOR_BGR2GRAY) 
                crop_img_resized = cv2.resize(gray_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
                #cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_rezized.jpg",crop_img_resized)
                return crop_img_resized,facedet_obj.left(),facedet_obj.top(),facedet_obj.right(),facedet_obj.bottom()
            else:
                return None,-1,-1,-1,-1
        else:
            return None,-1,-1,-1,-1
```
# Landmark detection and frontalization
A common problem in computer vision is the variety of transformations the images can have, they can be rotated at certain degree or they can have different perspectives. 

The next step after face detection is to extract the face landmarks, landmarks are special points in the face that relate to specific relevant parts like the jaw, nose, mouth and eyes, with the detected face and the landmark points it is possible to warp the face image to have a frontal version of it, luckily for us landmark extraction and frontalization can be simplified a lot by using the some dlib libraries.

//code fragment for landmark detector

```python
#landmark detector
shape = self.md_face(img,facedet_obj)
```

md_face receives the face region and will detect 68 landmark points using a previously trained model, with the landmark data we can make a warp transformation to the face using the landmarks as a guide to have the face image facing front.

//image of landmarks with sample face
![bengio_language_model.png]({{site.baseurl}}/assets/bengio_language_model.jpg)

to warp the face using the landmark data we use a python ported code that use the frontalization techinque by http://www.openu.ac.il/home/hassner/projects/frontalize/ ported by Heng Yang, the complete code can be found at the end of this post:

```python
p2d = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)], np.float32)
rawfront, symfront = self.fronter.frontalization(img,facedet_obj,p2d)
```

//image of face affined
![bengio_language_model.png]({{site.baseurl}}/assets/bengio_language_model.jpg)

# Image slicing
Now that we have complete frontal faces we can focus on the mouth region only:

```python
rawfront, symfront = self.fronter.frontalization(img,facedet_obj,p2d)
face_hog_mouth = symfront[165:220, 130:190] #get half-bottom part
```

//image of mouths parts
![bengio_language_model.png]({{site.baseurl}}/assets/bengio_language_model.jpg)

//picture of a bunch of mouths
![bengio_language_model.png]({{site.baseurl}}/assets/bengio_language_model.jpg)

With those transformations in place we can assure that our net will recive inputs of the same part of the face everytime without any additional noise improving our precision on the final model, the output of this step will be our true training data, finally!


# Histogram Equalization
A usefull technique for highlighting the details on the image is to apply histogram equalization:

```python
def histogram_equalization(img):
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img
```

# Data Augmentation
As you recall, we have labeled only 700 images from the MUCT database and 1500 from the LFW database, this is just not enought data for learning to detect teeths, we need to gather more data somehow, the obvious solution is to manual label a couple of thousands images in addition and this is the PREFERED solution, having more data is always better, but it is expensive in time, so for simplicity we are going to augment the data a pretty common technique in machine learning, specifically we are going to make the following transformations get almost 10x times more data:

* Mirroring of mouths
For each image we are going to create a mirrored clone, this will give us 2x the data.
//example of mirroring with a muct image

* Rotating of mouths
For each image we are going to make small rotations, specifically -30,-20,-10,+10,+20,+30 degrees of rotation this will give us 8x the data.

//rotating the image to create more data
```python
input_folder = "../img/mouth_data"
input_data_set = [img for img in glob.glob(input_folder+"/*jpg")]
output_folder ="../img/all_data"
generate_random_filename = 1

for in_idx, img_path in enumerate(input_data_set):
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    print(file_name)
    augmentation_number = 8
    initial_rot = -20
    path = output_folder+"/"+file_name+".jpg"
    copyfile(img_path, path)
    for x in range(1, augmentation_number):
        rotation_coeficient = x
        rotation_step=5
        total_rotation=initial_rot+rotation_step*rotation_coeficient
        mouth_rotated = image_rotated_cropped(img_path,total_rotation)
        #resize to 50 by 50
        mouth_rotated = cv2.resize(mouth_rotated, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
        if generate_random_filename == 1:
            guid = uuid.uuid4()
            uid_str = guid.urn
            str_guid = uid_str[9:]
            path = ""
            if 'showingteeth' in img_path:
                path = output_folder+"/"+str_guid+"_showingteeth.jpg"
            else:
                path = output_folder+"/"+str_guid+".jpg"
            cv2.imwrite(path,mouth_rotated)
        else:
            path = ""
            if 'showingteeth' in img_path:
                path = output_folder+"/"+file_name+"_rotated"+str(x)+"_showingteeth.jpg"
            else:
                path = output_folder+"/"+file_name+"_rotated"+str(x)+".jpg"
            cv2.imwrite(path,mouth_rotated)

```

# Seting up the Convolutional neural network in Caffe
The folling steps are requried to correctly configure a convolutional neural network in caffe:

# Preparing the training set and validation set
Now that we have mouth data ready, we need to split it into two subsets, we are going to use an 80/20 rule, 80 percent (15000 mouth images) of our transformed data is going to be the training set and the rest of the 20 percent (4500 mouth images) is going to be the validation set. The training data will be used during the training phase for learning and the validation set will be used to test the performance of the net during training, in this case, we have to move the mouth images to their respective folders located in training_data and validation_data

our training set folder
![pic](../images/trainingdata.jpg)

our validation set folder
![pic](../images/validationdatafolder.jpg)

# Creating the LMDB file

With the mouth images located in the training and validation folders, we are going to generate two text files, each containing the path of the corresponding mouth images plus the label (1 or 0), these text files are needed because Caffe has a tool to generate LMDB files based on these plain text files.

//code for generating those text files
```python
import caffe
import lmdb
import glob
import cv2
import uuid
from caffe.proto import caffe_pb2
import numpy as np
import os


train_lmdb = "../train_lmdb"
train_data = [img for img in glob.glob("../img/training_data/*jpg")]
val_data = [img for img in glob.glob("../img/validation_data/*jpg")]

myFile = open('../training_data.txt', 'w')

for in_idx, img_path in enumerate(train_data):
    head, tail = os.path.split(img_path)
    label = -1
    if 'showingteeth' in tail:
        label = 1
    else:
        label =0
    myFile.write(tail+" "+str(label)+"\n")

myFile.close()


f = open('../training_val_data.txt', 'w')

for in_idx, img_path in enumerate(val_data):
    head, tail = os.path.split(img_path)
    label = -1
    if 'showingteeth' in tail:
        label = 1
    else:
        label =0
    f.write(tail+" "+str(label)+"\n")

f.close()

```
//example of the file
![pic](../images/trainingdataplain.jpg)

To generate both training and validation LMDB files we run the following commands:

```bash
convert_imageset --gray --shuffle /devuser/Teeth/img/training_data/ training_data.txt train_lmdb
convert_imageset --gray --shuffle /devuser/Teeth/img/validation_data/ training_val_data.txt val_lmdb
```

# Extracting the mean data for the entire dataset
A common step in computer vision is to extract the mean data of the entire training dataset to facilitate the learning process during backpropagation, Caffe already has a library to calculate the mean data for us:

```bash
compute_image_mean -backend=lmdb train_lmdb mean.binaryproto
```

This will generate a file called mean.binaryproto, this file will have matrix data related to the overall mean of our training set, this will be subtracted during training to each and every one of our training examples to have a more reasonable scale for the inputs.

# Designing and implementing the convolutional neural network

Convnets are really good at image recognition because they can learn features automatically just by providing input and output data, they are also very good at transformation invariances this is small changes in rotation and full changes in translation.
In Machine Learning there are a set of well-known architectures for image processing like AlexNet, VGGNet, Google Inception etc. If you follow that kind of architectures is almost guaranteed you will obtain the best results possible, for this case and for the sake of simplicity we are going to use a simplified version of these nets with much less convolutional layers, remember that here we are just trying to extract Teeth features from the face and not entire concepts of the real world like AlexNet does, so a net with much less capacity will do fine for the task.

//code of 3 prototxt

train_val_feature_scaled.prototxt
```json
name: "LeNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
    mean_file: "mean.binaryproto"
    mirror: false
    
  }
  data_param {
    source: "train_lmdb"
    batch_size: 256
    backend: LMDB
  }
}

layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
    mean_file: "mean.binaryproto"
    mirror: false
  }
  data_param {
    source: "val_lmdb"
    batch_size: 256
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}

layer {
  name: "drop1"
  type: "Dropout"
  bottom: "ip1"
  top: "ip1"
  dropout_param {
    dropout_ratio: 0.5
  }
}

layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
```

deploy.prototxt

```json
name: "LeNet"
layer {
  name: "input"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 1
      dim: 100
      dim: 100
    }
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "drop1"
  type: "Dropout"
  bottom: "ip1"
  top: "ip1"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "softmax"
  type: "Softmax"
  bottom: "ip2"
  top: "pred"
}

```

solver.prototxt
```json
net: "model/train_val_feature_scaled.prototxt"
test_iter: 5
test_interval: 100
base_lr: 0.01
lr_policy: "step"
gamma: 0.1
stepsize: 500
display: 10
max_iter: 10000
momentum: 0.9
weight_decay: 0.0005
snapshot: 100
snapshot_prefix: "model_snapshot/snap_fe"
solver_mode: GPU
```

CNN Architecture
![pic](../images/architectureTeethCNN.png)

# Train the neural network

```bash
caffe train --solver=model/solver_feature_scaled.prototxt 2>&1 | tee logteeth_ult_fe_2.log
```

# Plotting loss vs iterations 
A good way to measure the performance of the learning in our convolutional neural network is to plot the loss on the training and validation set vs the number of iterations

//command to plot

//image of loss vs iterations with 10000 iterations
![bengio_language_model.png]({{site.baseurl}}/assets/bengio_language_model.jpg)

# Testing the trained model with unseen data

# Testing for a single image

First I'm going to test the net with some individual unseen images
Testing an individual image 
```python
mean_blob = caffe_pb2.BlobProto()
with open('../mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())

mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

mean_array = mean_array*0.003921568627

net = caffe.Net('../model/deploy.prototxt',1,weights='../model_snapshot/snap_fe_iter_2700.caffemodel')
net.blobs['data'].reshape(1,1, IMAGE_WIDTH, IMAGE_HEIGHT)  # image size is 227x227
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 0.00392156862745) 

img = cv2.imread(individual_test_image, cv2.IMREAD_UNCHANGED)
mouth_pre = mouth_detect_single(individual_test_image) #mouth is grayscale 1..255 50x50 BGR
if mouth_pre is not None:
    mouth_pre = mouth_pre[:,:,np.newaxis]
    mouth = transformer.preprocess('data', mouth_pre)
    net.blobs['data'].data[...] = mouth
    out = net.forward()
    pred = out['pred'].argmax()
    print(individual_test_image)
    print("Prediction:")
    print(pred)
    print("Prediction probabilities")
    print(out['pred'])
```

//image of samples images and probabilities

#Bulk testing
Testing the image by moving them to the correct folder
Now I'm going to test over an entire folder of unseen images, in this case, the folder called b have images taken on different angles so we can see is unseen data. Because I'm testing with a bunch of new data we need a way to measure the net performance

```python
accuracy = (true_negative + true_positive)/total_samples
recall = true_positive / (true_positive + false_negative)
precision = true_positive / (true_positive + false_positive)
f1score = 2*((precision*recall)/(precision+recall))
```

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Negative Cases | TN: 259            | FP: 26             |
| Positive Cases | FN: 80             | TP: 346            |

Total samples 751

The model looks good.

Now we have metrics to benchmark our trained model, with this in place we can quickly start tweaking things in our model or experimenting with different approaches and at the end see the final improvement with a number.

# Testing our net with real video!
Although training a convnet is a very slow process, testing it is not!, in fact, it takes milliseconds to test the trained model, to prove you that I'm going to call the trained net in each frame of a video to show the predictions on realtime. 

```python
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    result = predict(frame)
    if(result == 1):
        size = cv2.getTextSize("Showing teeth", cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
        x,y = (50,250)
        label_top_left = (x - size[0]/2, y - size[1]/2)
        print(frame)
        cv2.rectangle(frame, (x,y),(x+size[0],y-size[1]),(0,255,0),-2);
        cv2.putText(frame, "Showing teeth",(x,y),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0))
```

//gif of a video running with the net showing if the person is showing the teeth or not
![bengio_language_model.png]({{site.baseurl}}/assets/bengio_language_model.jpg)
