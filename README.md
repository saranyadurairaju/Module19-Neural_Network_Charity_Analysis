# Charity data Analysis with Neural Network

Neural networks are computing systems with interconnected nodes that work much like neurons in the human brain. Using algorithms, they can recognize hidden patterns and correlations in raw data, cluster and classify it, and – over time – continuously learn and improve. Deep learning is a deep neural network with many hidden layers and many nodes in every hidden layer. Deep learning develops deep learning algorithms that can be used to train complex data and predict the output.

![image](https://user-images.githubusercontent.com/85472349/139388057-e1fc5e2f-ae71-4306-949e-6a5fe2bf1315.png)


## Overview

The purpose of this project is to use deep-learning neural networks with the TensorFlow platform in Python, to analyze, classify and optimize the success of charitable donations. We use the following methods for the analysis:

* Deliverable 1: Preprocessing Data for a Neural Network Model
* Deliverable 2: Compile, Train, and Evaluate the Model
* Deliverable 3: Optimize the Model

## Resources

•	Data Source: [charity_data.csv]

•	Software: Python, Anaconda Navigator, Conda, Jupyter Notebook

## Analysis and Results

We will use jupyter notebook with python packages for our Analysis. First we will import the dependencies, define the columns, Load the data. Then we will follow few steps to optimize our model. Below are the deliverable details:

### Deliverable 1: Preprocessing Data for a Neural Network Model 

With our knowledge of Pandas and the Scikit-Learn’s StandardScaler(), we’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable. The following preprocessing steps have been performed:

* The EIN and NAME columns have been dropped
* Columns with more than 10 unique values have been grouped together
* Categorical variables have been encoded using one-hot encoding


![image](https://user-images.githubusercontent.com/85472349/139452116-6ae8bc4d-0fd3-4df5-963b-f2a8ce795d12.png)

* Preprocessed data is split into features and target arrays 
* Preprocessed data is split into training and testing datasets 
* Numerical values have been standardized using the StandardScaler() module

![image](https://user-images.githubusercontent.com/85472349/139452289-a5ee6112-c9a5-4aa6-8f9a-9f660e1f5500.png)


![image](https://user-images.githubusercontent.com/85472349/139452211-4d5ae8ef-5446-49f1-a727-798add67d7ec.png)

### Deliverable 2: Compile, Train, and Evaluate the Model 

With our knowledge of TensorFlow, we’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. We’ll need to think about how many inputs there are before determining the number of neurons and layers in our model. Once we’ve completed that step, we’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

The neural network model using Tensorflow Keras contains working code that performs the following steps:

 * The number of layers, the number of neurons per layer, and activation function are defined 
 * An output layer with an activation function is created
 * There is an output for the structure of the model
 * There is an output of the model’s loss and accuracy 
 * The model's weights are saved every 5 epochs

![image](https://user-images.githubusercontent.com/85472349/139454790-4261dbca-83c6-4e14-9f7f-01e52caf318b.png)

**Neural network model using Tensorflow Keras**

hidden_nodes_layer1 =  80

hidden_nodes_layer2 = 30

Activation function (First hidden layer)  = relu

Activation function (Second hidden layer)  = relu

Activation function (Output layer)  = sigmoid

epochs = 100

loss: 0.5355 - accuracy: 0.7411
 
**Evaluation of Model Test data**

Loss: 0.5556238889694214, Accuracy: 0.7254810333251953

**With epoch reduction:**

epochs = 30

loss: 0.5342 - accuracy: 0.7423

**Evaluation of Model Test data**

Loss: 0.5556238889694214, Accuracy: 0.7254810333251953

**The results are saved to an HDF5 file**
 
 ![image](https://user-images.githubusercontent.com/85472349/139456052-a24db45a-500b-47bb-921d-d026b7fc3591.png)


### Deliverable 3: Optimize the Model

With our knowledge of TensorFlow, optimize our model in order to achieve a target predictive accuracy higher than 75%. If we can't achieve an accuracy higher than 75%, we'll try different attempts.


### Adding more neurons to a hidden layer

![image](https://user-images.githubusercontent.com/85472349/139519987-99148902-779f-48c1-939c-439465e19783.png)

![image](https://user-images.githubusercontent.com/85472349/139520053-d3d367ed-75ee-4212-85d5-29e598cb9f33.png)


### Adding more hidden layers

![image](https://user-images.githubusercontent.com/85472349/139520069-71788570-9b1b-4511-9613-e00664de4a71.png)

![image](https://user-images.githubusercontent.com/85472349/139520075-0bfb35a9-8872-4c52-a8d9-8eb688fdd695.png)


### Using different activation functions (tanh) for the hidden layers

![image](https://user-images.githubusercontent.com/85472349/139520112-10abaa91-9f49-4355-b8a8-f50f5e6508b5.png)

![image](https://user-images.githubusercontent.com/85472349/139520119-4e5cbf0a-e4f1-4afb-9720-89adb9404dc6.png)


### Using different activation functions for both the hidden layers with increased epoch

![image](https://user-images.githubusercontent.com/85472349/139520164-69eac233-4af7-4a79-ac0c-992bb3c626ec.png)

![image](https://user-images.githubusercontent.com/85472349/139520181-42ae9ed8-5250-4430-a6ef-e480ecec8720.png)


### Adding the number of epochs to the training regimen

![image](https://user-images.githubusercontent.com/85472349/139520210-2f7effa5-83bc-4c9d-a5c8-76e5d8595890.png)

![image](https://user-images.githubusercontent.com/85472349/139520218-28799fcf-08e1-4f6e-acb1-492e82f7611a.png)


### Reducing the number of epochs to the training regimen

![image](https://user-images.githubusercontent.com/85472349/139520697-115d7508-45e5-481b-ba6c-6595b2360131.png)


### Export the module to HDF5 file

![image](https://user-images.githubusercontent.com/85472349/139520721-499cd715-b0b3-49a2-8e43-1ded5635bc46.png)


### Random Forest Classifier

![image](https://user-images.githubusercontent.com/85472349/139520761-8d6ca184-607c-4919-8411-719c2714a983.png)


## Summary

The models accuracy ended up around 72% in all scenarios. From the original data set we dropped few irrelevant columns, combined the values and then we tried neural network on different scenarios. We also tried using Random Forest Classifier. Still we ended up with the same level of Accuracy. 

So, either we need to alter our dataset by removing few items or we need to have more data to get a expected result. Finally having the proper data is very important for any modelling. 
