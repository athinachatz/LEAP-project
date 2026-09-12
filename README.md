# LEAP-project
**Handwritten Digit Recognition with CNNs in Python (MNIST Dataset)**

The goal of this project was to build and train a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. Beyond evaluating overall accuracy, the project focuses on analyzing the model's specific misclassifications.

## Data Set 
The MNIST data set consists of 70,000 images of handwritten digits, labelled 0 to 9. Each image is 28 pixels by 28 pixels in greyscale.The data was downloaded from OpenML (an open-source machine learning platform) and initially loaded as an array of 70,000 rows, where each row contained 784 numbers representing a flattened image. 

## Model 
1. Enviroment Setup
     This notebook was developed and tested in Google Colab. The project utilizes NumPy for array manipulation, Matplotlib for visual plotting, and Scikit-learn for dataset preprocessing. The neural network was built and trained using TensorFlow, with Keras configured as the backend.
2. Preprocessing
     To prepare the inputs for the Convolutional Neural Network, these flat rows were reshaped back into 28x28 grids with a single grayscale channel. Finally, the raw pixel values were scaled from a 0-255 range down to 0-1 to help the network train more efficiently.The data is then split into a training set (80%) and a test set (20%), keeping the class balance. The digits' labels are converted to integers and saved in an array so the model can work with them. 

3. Model
   The network is built using the Sequential model (straight stack of layers, added in order). 

Input layer : Expects an incoming matrix of (28, 28, 1), representing a 28x28 pixel image with a single grayscale color channel. 

Convolutional layer(Conv2D) : Applies 32 filters  with a 3x3 window and a ReLU activation function to extract spatial feaures (like edges and curves) from digits.

Pooling layer(MaxPooling2D) : Uses a 2x2 window to downsample the feature maps, reducing computational load.

Dropout layer : Randomly deactivates 30% of neurons during training to prevent overfitting.

Flatten layer : Converts the 2D matrix into a 1D array so it can be passed into the traditional neural network layers.

Dense hidden layer : Fully connected layer with 64 units and a ReLU activation function that learns complex patterns.

Dense output layer : Fully connected layer with 10 units (one unit per class) using softmax activation function to output the probability for each class.


   
