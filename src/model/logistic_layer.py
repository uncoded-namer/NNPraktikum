
import time

import numpy as np

from util.activation_functions import Activation

class LogisticLayer:
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='softmax', isClassifierLayer=True, learningRate=0.02):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.derivative = Activation.getDerivative(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input    = np.ndarray((nIn+1, 1))
        self.input[0] = 1
        self.output   = np.ndarray((nOut,  1))
        self.delta    = np.zeros((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1))-0.5
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size  = self.nOut
        self.shape = self.weights.shape
        
        self.learningRate = learningRate
        self.raw_out      = None

    def forward(self, input):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        
        # self.input   = input
        self.input   = np.reshape(input, (input.shape[0], 1))
        self.raw_out = np.dot(self.weights, self.input)
        
        return self.activation(self.raw_out)

    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """
        
        #if type(self.raw_out) == type(None):
        #    raise Error("LogisticLayer.computeDerivative() called before LogisticLayer.forward()")
        
        self.delta = self.derivative(self.raw_out) * nextDerivatives.dot(nextWeights)

    def updateWeights(self):
        """
        Update the weights of the layer
        """
        
        #if len(self.delta) != self.nOut or len(self.input) != (self.nIn + 1):
        #    raise Error("Type confusion!")
        
        self.weights = self.weights + np.dot(self.delta, self.input.T)
