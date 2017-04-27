# import the necessary modules for your ANN below:
# For instance: import numpy as np
import numpy as np

class ANN(object):
    def __init__(self, num_inputs, num_hidden_nodes,num_layers, num_outputs, weights):
        self.weights = weights # this is an individual which is actually a list of weights (genome)
        # You need to extract the list of weights and put each weight into correct place in your ANN.
        self.num_layers = num_layers;
        self.num_hidden_nodes = num_hidden_nodes
        # Therefore ANN topology should be fixed through all generations.
        self.num_hidden_weights = (num_inputs+1) * num_hidden_nodes
        self.num_layer_weights = (num_layers - 1) * (num_hidden_nodes+1) * num_hidden_nodes

        # Let's assume that you have one hidden layer, then you would end up with two matrices of weights:
        # 1) Between the input layer and the hidden layer
        # 2) Between the hidden layer and the output layer
        # Hence, place them in the following hidden and output weights variables, respectively.
        #print weights
        self.hidden_weights = np.array(weights[:self.num_hidden_weights]).reshape(num_hidden_nodes,num_inputs+1)
        self.layer_weights =  weights[self.num_hidden_weights:self.num_hidden_weights + self.num_layer_weights];
        self.output_weights = np.array(weights[(self.num_hidden_weights + self.num_layer_weights):]).reshape(num_outputs,num_hidden_nodes+1)
    
    def activation(self,x):
        # x is the net input to the neuron (previously represented as "z" during the class)
        # a is the activation value ( a = activation(z) )
        # activation function could be sigmoid function: 1/(1+exp(-x))
        a = 1/(1+np.exp(-x))
        return a

    def evaluate(self,inputs):
        # Compute outputs from the fully connected feed-forward ANN:
        # So basically, you will perform the operations that you did on HW4:
        # Let's assume that you have one hidden layer with 2 hidden nodes. Then you would have
        # a matrix of weights (first layer of weights beetween the input and hidden layers) of
        # size: 2 x (4+1)=2 x 10, and another matrix of weights (second layer between the hidden
        # layer and the output layer) of size: 2 x (2+1) = 6, resulting in total of 16 weights.
        # First, compute z(2) vector which is 2-by-1:
        # z(2,1) = +1 x weight[0] + x(1) x weight[1] + ... + x(4) x weight[4]
        # z(2,2) = +1 x weight[5] + x(1) x weight[6] + ... + x(4) x weight[9]
        # a(2,1) = activation(z(2,1))
        # a(2,2) = activation(z(2,2))
        # Second, compute z(3) as similar to z(2). This time, the inputs are the outputs of the first layer (i.e., a(2))
        # z(3,1) = ...
        # z(3,2) = ...
        # a(3,1) = ... = outputs(1) = velocity_left
        # a(3,2) = ... = outputs(2) = velocity_right
        # Output the results (left_track_speed and right_track_speed) as a 
        
        inputs.insert(0,1)
        layer = []
        for i in range(len(self.hidden_weights)):
            z = 0
            for j in range(len(inputs)):
                z += self.hidden_weights[i][j] * inputs[j]
            layer.append(self.activation(z)) 

        oldLayer = layer
        newLayer = []
        for c in range(self.num_layers - 1):
            oldLayer.insert(0,1)
            newLayer = []
            layerWeights = len(oldLayer) * (len(oldLayer)-1);
            thisWeights = np.array(self.layer_weights[layerWeights * c:layerWeights * (c+1)]).reshape(self.num_hidden_nodes,self.num_hidden_nodes+1)
            for i in range(len(thisWeights)):
                z = 0
                for j in range(len(oldLayer)):
                    z += thisWeights[i][j] * oldLayer[j]
                newLayer.append(self.activation(z)) 
            oldLayer = newLayer
        layer = oldLayer
        outputs = []
        for i in range(len(self.output_weights)):
            z = 0
            for j in range(len(layer)):
                z += self.output_weights[i][j] * layer[j]
            outputs.append(self.activation(z))

        return outputs
        

