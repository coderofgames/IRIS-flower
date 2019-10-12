========================================================================
    CONSOLE APPLICATION : ArtificialNeuralNetwork Project Overview
========================================================================

NOTE: This sample will fry your brains 

NOTE: *Update* starting to implement a solution to the dissappearing gradient
problem. There are multiple versions of the layered neural network in 
their own namepsaces because I need to keep hopping back to previous versions

The Sample currently classifies the spiral dataset (see the ipynb) with 99%
accuracy - I have tuned the learning rate of each individual layer.

NOTE: after recent Matrix optimizations and improvements, the classifier is finishing in 72%
of the time in the shallow test, and 77% of the time in the deep test (in debug mode - better still, 
build in release mode)

An extension of the matrix form of Artificial Neural Network employed as a classifier 
(after much painstaking research and error checking for the correct form of matrix multiplication)
This example classifies the Iris flower, one of the example CSV (comma seperated value) files that
comes built in to Scikit-learn in Scipy (scientific python).

The flowers were classified with the following input data
sepal length
sepal width
petal length
petal width
so given that information you can discriminate a type easily. 

The neural network layer class is currently working for a deep calculation
of a network with 2 hidden layers and 1 output layer, and also still
works for a network with 1 hidden layer and 1 output layer. 

Special Thanks to:
https://trac.xapian.org/attachment/ticket/598/ann.cpp
for how to account for the error in the data

http://briandolhansky.com/blog/2014/10/30/artificial-neural-networks-matrix-form-part-5
for a helpful chapter on the matrix method of ANN's, although I could not convert the 
python code to C++

and last but not least ...
Michael A. Nielsen, "Neural Networks and Deep Learning"
online book
http://neuralnetworksanddeeplearning.com/chap2.html
