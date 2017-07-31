
import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score
import sys
import logging
import copy
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='ce', learningRate=0.001, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation

        self.trainingSet = copy.deepcopy(train)
        self.validationSet = copy.deepcopy(valid)
        self.testSet = copy.deepcopy(test)

        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        elif loss == 'ce':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        outputActivation = self.outputActivation

        self.inputWeights = inputWeights
        if layers is None:
            self.layers.append(LogisticLayer(nIn=train.input.shape[1],
                                             nOut=128,
                                             weights=self.inputWeights,
                                             activation=inputActivation,
                                             isClassifierLayer=False))
            # Output layer
            self.layers.append(LogisticLayer(nIn=128,
                                             nOut=10,
                                             weights=None,
                                             activation=outputActivation,
                                             isClassifierLayer=True))
        else:
            self.layers = layers




        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                           axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                             axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """

        inp = self._get_input_layer().forward(inp)
        for layer in self.layers[1:]:
            inp = np.insert(inp, 0, 1, axis=0)
            inp = layer.forward(inp)
        return inp
        
    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        newTarget = np.zeros(10)
        newTarget[target] = 1

        for layer in reversed(self.layers):
            if layer.isClassifierLayer:
                derivates = - self.loss.calculateDerivative(newTarget, layer.outp)
                weights = np.ones(layer.shape[1])

            layer.computeDerivative(derivates, weights)
            derivates = layer.deltas
            weights = layer.weights[1:]

    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(learningRate)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        learned = False
        iteration = 1
        while not learned:
            for data, label in zip(self.trainingSet.input, self.trainingSet.label):
                self._feed_forward(data)
                self._compute_error(label)
                self._update_weights(self.learningRate)
            accuracy = accuracy_score(self.validationSet.label,
                                      self.evaluate(self.validationSet))
            self.performances.append(accuracy)
            if verbose:
                logging.info("Epoch: %i/%i; Accuracy: %f", iteration, self.epochs, accuracy * 100)

            if accuracy == 1.0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True
            iteration += 1



    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        output = self._feed_forward(test_instance)
        return np.argmax(output)
        

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
