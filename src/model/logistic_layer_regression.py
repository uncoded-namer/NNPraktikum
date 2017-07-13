# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from util.loss_functions import DifferentError, BinaryCrossEntropyError, MeanSquaredError

from model.classifier import Classifier
from model.logistic_layer import LogisticLayer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticLayerRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.005, epochs=30):
        
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        nIn  = train.input.shape[1]
        nOut = 1
        
        self.layer = LogisticLayer(nIn, nOut, weights=None, activation='sigmoid',
                                   isClassifierLayer=True, learningRate=learningRate)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        
        loss = MeanSquaredError()

        learned = False
        iteration = 0

        while not learned:
            
            # MSE
            totalError = 0
            
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                
                vec     = (np.array([1]), input)
                output  = self.layer.forward(np.concatenate(vec))
                
                totalError = totalError + loss.calculateError(label, output)
                
                self.layer.computeDerivative(label - output, np.array([1]))
                self.layer.updateWeights()
                
            totalError = abs(totalError)
            iteration += 1
        
            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)
        
            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        
        cc = np.insert(testInstance,0,[1])
        r  = self.layer.forward(cc)[0][0] > 0.5
        
        return r

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
