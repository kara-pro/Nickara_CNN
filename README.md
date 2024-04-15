Model Description:

Name: Nickara_MNIST_Model
Type: Convolutional Neural Network (CNN)
Framework: Keras
Purpose: This model can be used to identify handwritten digits. The model was trained using data augmentation, making it more resilient against imperfect data.

Data Augmentation and Splitting
Our original dataset is the Keras MNIST dataset. However, we used ImageDataGenerator to augment the images. Our settings are as follows:
   ImageDataGenerator(width_shift_range=.2, height_shift_range=.2, rotation_range=15, fill_mode='constant',rescale=1./255, zoom_range=[.75,2])

   This shifts the images horizontally and vertically, rotates them a little, zoomins in and out of them, and recales the images to grayscale.

We split this into train and validation data sets with a test_size of 0.2, shuffle=True, and stratifying on the labels.


Model Architecture

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

Activation Functions:
   - ReLU (Rectified Linear Unit) in hidden layers
   - Softmax in the output layer

Output Layer:
   - Dense layer with 10 units (number of classes) and softmax activation function

Training Procedure:
   - Optimization Algorithm: SDG
   - Loss Function: Categorical Cross-Entropy
   - Regularization: Dropout (rate=0.2)

Model Performance
   - Validation Accuracy: 0.85
   - Precision: 0.86
   - Recall: 0.84
   - F1-score: 0.85

##Steps for MLFlow Integration
For MLFLow integration, we used it in conjunction with HyperOpt. Our search space was on the dropout rate, activation functions for the convolution layers, and the size of the convolution layers. We did 5 evalutions and found that the above architecture without the additional dropouts after the convolution layers performed the best. However, by adding more dropouts we got better results.

We registered the best model in MLFlow and integrated it with APIFlow

##Steps for API usage
1. Download the docker container
2. Use the following commands:
   docker build -t Nickara_MNIST_Container .
   docker run -p 8000:8000 Nickara_MNIST_Container

##Deployment Process
