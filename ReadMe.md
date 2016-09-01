========================================================================
    CONSOLE APPLICATION : ArtificialNeuralNetwork Project Overview
========================================================================

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

The neural network layer class does not neccessarily work for all
possible cases because I have not tested all possible cases, for example
one bug in the last implementation (in the ANN project) was that the
Hadamard product operator precedence was wrong, but since I was only
computing the result of an exclusive OR gate the matrix calculations still
worked since the matrix of size 1*1 degenerates to multiplication by a scalar.


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