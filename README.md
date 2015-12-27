# Neural-Network

First attempts at implementing a neural network. The network is fully connected. 
Data present in the input nodes is normalized (1/max). Nodes in hidden layers are using sigmoid activation. The output layer is using softmax.

##Examples.

Creates a three layered network with two input nodes, three hidden nodes and two output nodes:

'''C++
Network n({2, 3, 2}); 
'''

Creates a four layered network with 784 input nodes, 128 nodes in the first hidden layer, 63 nodes in the second hidden layer and and 10  output nodes:
'''C++
Network n({784, 128, 63, 10});
'''

Load a dataset to be used for training or validation. The custom data set consist of two lists of comma separated numbers for input and expected output. The lists are separated by "->". A leading '#' indicates a comment.

'''C++
# Use a 5:5:1 network

1,1,0,0,0 -> 1,0
0,1,1,0,0 -> 1,0
0,0,1,1,0 -> 1,0
0,0,0,1,1 -> 1,0
1,0,0,0,1 -> 1,0
0,0,0,0,0 -> 0,1
1,1,1,0,0 -> 0,1
0,1,1,1,0 -> 0,1
'''

Creating a data set:

'''C++
    // The MNISTDataSet is a specialized DataSet that can load from  binary MNIST data.
    MNISTDataSet trainingSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    MNISTDataSet testSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    
    // Custom data set. Described above.
    CustomDataSet custom("XOR.training");
'''

An epoch is defined as one pass through all examples in the training set. Calling the "Train" method make the network start loading 
examples from the training set, normalize the input, feed forward, compare outputs to expected output and then back propagate the error 
in order to adjust weights and biases.
The "Train" method will exit if it reaches maxEpoch or converges to a solution before it has reached maxEpoch.

'''C++
  int maxEpoch = 50000;
  double learningRate = 0.2;
  
  int epoch = n.Train(trainingSet, learningRate, maxEpoch);
'''

After training on the training set, performance can be tested against the test set, bias vectors and weight matrixes can be saved to file. The network can also serialize to a digraph representation that can be viewed in GraphViz:

'''C++
  n.Run(testSet); // Runs the trained network on a dataset.
  n.ExportAsDigraph("d:\\network.gv"); // Export to graphviz
  n.Serialize("d:\\MNISTWeightsAndBiases.network"); // Persist biases and weights
'''

PS. This is a first attempt (== naive implementation). It may contain mathematical errors and is probably not optimal in regards to speed / performance.

##Further work: 
- batch training / randomizing training examples
- mechanisms to prevent overfitting
- training set or weight parallelization ?
- ...

![License](http://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-sa.png)

Hans Jørgen Grimstad
www.TimeExpander.com
