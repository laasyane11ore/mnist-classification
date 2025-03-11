# MNIST Digital Classification with Tensorflow
## Overview
This project builds a handwritten digit classifier using the MNIST dataset and a simple neural network implemented with TensorFlow and Keras. The model is trained on grayscale 28x28 images of digits(0-9) and tested for accuracy in predicting handwritten numbers.
## Dataset
The MNIST dataset contains:<br>
* 60,000 training images 
* 10,000 test images <br>
Each image is 28x28 pixels and labeled from 0-9.
## Implementation
1. Load the dataset using tf.keras.datasets.mnist.load.data()
2. Normalize the pixel values
3. Define the neural network
* Flatten layer converts images into 1D array
* A fully connected hidden layer - Dense(128, activation = 'relu')
* Prevent overfitting - Dropout(0.2)
* Output probabilities for 10 digit classes - Dense(10, activation = 'softmax')
4. Use Adam optimizer and categorical cross-entropy loss
5. Train for 10 epochs with validation on the test dataset
6. Evaluate accuracy and loss on test data
7. Make predictions on a random test image
8. Save the model and reload it for further evaluation

## Expected Results
The model achieves a test accuracy of 97% to 98% making it effective for digit recognition

## Potential Improvements
1. Using CNN for better feature extraction.
2. Adding more hidden layers and experimenting with different activation functions.
3. Implementing data augmentation to improve generalization.

## Author
Laasya Nellore <br>
Github : https://github.com/laasyane11ore
