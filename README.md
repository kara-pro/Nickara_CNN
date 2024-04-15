<pre>
**Model Description:**

Name: Nickara_MNIST_Model
Type: Convolutional Neural Network (CNN)
Framework: Keras
Purpose: This model can be used to identify handwritten digits. The model was trained using data augmentation, making it more resilient against imperfect data.
Disclaimer: The model in the docker container is not the updated version. The model in this repository has an                    accuracy of 98.86% and the model in the docker container has an accuracy of 96.66%

Data Augmentation and Splitting
Our original dataset is the Keras MNIST dataset. However, we used ImageDataGenerator to augment the images. Our settings are as follows:
   ImageDataGenerator(width_shift_range=.2, height_shift_range=.2, rotation_range=15, fill_mode='constant',rescale=1./255, zoom_range=[.75,2])

   This shifts the images horizontally and vertically, rotates them a little, zooms in and out of them, and recales the images to grayscale.

We split this into train and validation data sets with a test_size of 0.2, shuffle=True, and stratifying on the labels.


**Model Architecture**

Our model has the following layers:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ CL1 (Conv2D)                         │ (None, 26, 26, 128)         │           1,280 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 26, 26, 128)         │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ MPL1 (MaxPooling2D)                  │ (None, 13, 13, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 13, 13, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ CL2 (Conv2D)                         │ (None, 11, 11, 64)          │          73,792 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 11, 11, 64)          │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ MPL2 (MaxPooling2D)                  │ (None, 5, 5, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 5, 5, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ FL (Flatten)                         │ (None, 1600)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │         102,464 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 16)                  │             528 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ OUTL (Dense)                         │ (None, 10)                  │             650 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 361,782 (1.38 MB)
 Trainable params: 180,698 (705.85 KB)
 Non-trainable params: 384 (1.50 KB)
 Optimizer params: 180,700 (705.86 KB)

_Activation Functions:_
   - ReLU (Rectified Linear Unit) in hidden layers
   - Softmax in the output layer

_Output Layer:_
   - Dense layer with 10 units (number of classes) and softmax activation function

_Training Procedure:_
   - Optimization Algorithm: SDG
   - Loss Function: Categorical Cross-Entropy
   - Regularization: Dropout (rate=0.2)

_Model Performance_
   - Validation Accuracy: 0.9886
   
_Confusion Matrix_
 [ 969    1    1    0    1    0    1    0    0    0]
 [   0 1129    0    0    0    0    3    1    1    0]
 [   2    2 1025    0    0    0    0    8    1    0]
 [   0    1    2 1000    0    2    0    1    7    0]
 [   0    0    1    0  977    0    2    4    1    8]
 [   3    0    0    9    0  889    4    0    1   11]
 [   3    0    0    0    0    1  947    0    2    0]
 [   2    2    3    1    0    0    0 1012    1    4]
 [   1    0    0    0    1    0    1    0  955    2]
 [   0    0    0    0    3    0    0    2    5  984]

##Steps for MLFlow Integration
For MLFLow integration, we used it in conjunction with HyperOpt. Our search space was on the dropout rate, activation functions for the convolution layers, and the size of the convolution layers. We did 5 evalutions and found that the above architecture without the additional dropouts after the convolution layers performed the best. However, by adding more dropouts we got better results.

We registered the best model in MLFlow and integrated it with FastAPI

##Deployment Process
To build the Docker instance, we had the requirements, model files logged with MLFlow locally, dockerfile, and app code in a folder. 
We ran the following commands to build and run the image:
   docker build -t myapp .
   docker run -p 8000:8000 myapp
As a user, you can do the following:
1. Download the myapp and open a terminal on your machine
2. Use docker start to activate your instance of Docker and run 'docker ps -a' to verify you have downloaded the container
3. Enter 'docker run -p 8000:8000 myapp' to start container and the image

##Steps for API usage
1. Follow the above steps to run the docker image
2. Use Postman to create a post request to 'http://127.0.0.1:8000/predict'
3. Put your data as a raw body in the following format:
   {"data": *your (28, 28, 1) array*}


##
</pre>
