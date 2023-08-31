
<h2>Introduction:</h2>
The development of neural networks, a subset of machine learning technology, is a valuable tool that many industries around the world are using in order to maximize the efficiency of their functionality. The spread of the usage of neural networks around the world has been key to the emergence of new technologies. One of these technologies is the advent of self-driving vehicles, which, if successfully implemented, would be able to improve the safety and efficiency of personal and commercial transportation. A self-driving vehicle, however, has to have the ability to detect the location and type of street signs in its view cone, and the best way to create a system which does this is through a neural network, which is what this repository hopes to provide. A convolutional neural network model which is able to detect 4 classes of common road signage, while also giving the coordinates of its bounding box within the frame. This model, and models similar to it, are able to overcome a major hurdle with the usage of self-driving vehicles. Training these models is able to efficiently create a system which can detect signage and facilitate the usage of these vehicles.


<h2>Technical Background:</h2>
<h3>What is a Neural Network?</h3>

A neural network is a type of artificial intelligence system, which uses layers of neurons or nodes, and trains them, in order to simulate the workings of a real life brain. To break it down, a neural network contains multiple layers of nodes, which connect to nodes in other layers. The starting layer, or the input layer, takes the initial raw data, in our case, the details of the image, with each node representing a feature of the input data. These nodes are connected to layers in between, called hidden layers, which process the information from the prior layers and pass it on to the next hidden or output layer. Often, pattern detection arises through the hidden layers. These outputs are finally transferred to the output layer, which gives an answer based on the strength of the nodes in this layer. Node strength is on a scale from 0.0 to 1.0, and each node might represent a different class. The strongest node is what the NN determined the object to be. Additionally, the confidence that NN has in its decision is based on the strength of the final answer node. 

<h3>What is a Convolutional Neural Network?</h3>

A Convolutional Neural Network, or a CNN, is a specialized type of neural network whose primary purpose is image recognition. The features that differentiate a CNN from other neural networks are the types of layers it uses. Firstly, the convolutional layer is a layer whose main purpose is developing patterns, such as edges, corners, or textures in an image. This is done through the usage of sliding “filters” over the image, which only looks at small portions of the picture at a time. There are other types of layers which are utilized in CNNs, which help maximize the efficiency of the convolutional layer, such as pooling layers, which reduce the computational complexity of images. 

<h3>How do you train a Neural Network model?</h3>

All nodes in a neural network have biases, which either increase or decrease the overall strength of that node. Additionally, all connections between nodes, or edges, have weights, which increase or decrease the chance of that connection being made. When training a neural network model, these biases and weights are randomized. Additionally, the datasets are split into training data and testing data. Training data is used during each cycle of training, or epoch, to find the ‘loss' of the model, which is used to find the gradient, or the necessary change, to each node’s weight and bias. This is done through a process known as backpropagation. This is then repeated through all the training epochs in order to produce a fully fledged neural network.


<h2>Literature Review:</h2>

**Efficient Model for Image Classification With Regularization Tricks:** This writeup, Taehyeon Kim, Jonghyup Kim, Seyoung Yun hopes to answer questions about the training of CNN models, and how we can make it as efficient as possible. The models outlined in the writing uses the CIFAR-100 dataset, an incredibly popular dataset with over 100 classes of objects of many household objects. It explores different methods of training CNNs in order to improve the accuracy of the final model

**Advancements in Image Classification using Convolutional Neural Network:** In this paper by Farhana Sultana, A. Sufian, and Paramartha Dutta, comparisons between different architectures of CNN models are drawn, hoping to evaluate which is the best to use. They used several image recognition focused datasets and used multiple popular and common CNN architectures. It also touches on the training techniques used, and how that influenced the overall success rate of the models.
 
**Deep Learning for Large-Scale Traffic-Sign Detection and Recognition:** This paper by Domen Tabernik and Danijel Skoca outline the processes used in a neural network with a goal similar to ours, that being to detect and classify several classes of street signage. They make use of a CNN model which is able to differentiate between 200 sign categories, including many difficult classes that were not previously considered in other works. The techniques used in the paper helped the progress of this project.

**VSSA-NET: Vertical Spatial Sequence Attention Network for Traffic Sign Detection:** This paper outline the techniques used in the development of a novel CNN model, by Yuan Yuan, Zhitong Xiong, and Qi Wang, focus on the intricacies of street sign classification by a neural network. It takes into account the difficulties of detecting small street objects and distinguishing between false targets. They use a variety of novel techniques in their model to account for these difficulties. 


<h2>Technical Tools:</h2>
The tools used in this model are primarily the Keras library, included with TensorFlow. This gave many of the necessary features and methods required to create, train, and implement CNN models, such as the layer types, training methods, etc. Also used was Element Tree, a Python library which allowed the code to parse through the .xml files, and Scikit-learn, a library which allowed the randomized split of the training and testing datasets. Numpy was also used.

The dataset used is “Road Sign Detection” posted by Larxel on Kaggle, which contains 4 classes of road signs with 877 images total, alongside their .xml files. 
https://www.kaggle.com/datasets/andrewmvd/road-sign-detection 


