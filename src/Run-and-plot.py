#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator

from matplotlib import pyplot as plt

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)


    params = {'legend.fontsize': 8}
    plt.rcParams.update(params)

    epochs_list = [100, 50, 20, 10]
    subplot_num = 1
    
    for epochs in epochs_list:
        plt.subplot(2, 2, subplot_num)
        subplot_num = subplot_num + 1
    
        learningRates = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        stupidRes     = []
        perceptronRes = []
        regressionRes = []
        
        for learningRate in learningRates:
        
            myStupidClassifier = StupidRecognizer(data.trainingSet,
                                                data.validationSet,
                                                data.testSet)
            
            myPerceptronClassifier = Perceptron(data.trainingSet,
                                                data.validationSet,
                                                data.testSet,
                                                learningRate=learningRate,
                                                epochs=epochs)
            
            myLogisticRegressionClassifier = LogisticRegression(data.trainingSet,
                                                                data.validationSet,
                                                                data.testSet,
                                                                learningRate=learningRate,
                                                                epochs=epochs)

            # Train the classifiers
            print("=========================")
            print("Training..")

            print("\nStupid Classifier has been training..")
            myStupidClassifier.train()
            print("Done..")

            print("\nPerceptron has been training..")
            myPerceptronClassifier.train()

            print("\nLogistic Regression has been training..")
            myLogisticRegressionClassifier.train()

            print("Done..")

            
            # Do the recognizer
            # Explicitly specify the test set to be evaluated
            stupidPred = myStupidClassifier.evaluate()
            perceptronPred = myPerceptronClassifier.evaluate()
            logisticRegressionPred = myLogisticRegressionClassifier.evaluate()

            # Report the result
            print("=========================")
            evaluator = Evaluator()

            print("Result of the stupid recognizer:")
            # evaluator.printComparison(data.testSet, stupidPred)
            evaluator.printAccuracy(data.testSet, stupidPred)
            stupidRes.append(evaluator.getAccuracy(data.testSet, stupidPred))

            print("\nResult of the Perceptron recognizer:")
            # evaluator.printComparison(data.testSet, perceptronPred)
            evaluator.printAccuracy(data.testSet, perceptronPred)
            perceptronRes.append(evaluator.getAccuracy(data.testSet, perceptronPred))

            print("\nResult of Logistic Regression recognizer:")
            evaluator.printAccuracy(data.testSet, logisticRegressionPred)
            regressionRes.append(evaluator.getAccuracy(data.testSet, logisticRegressionPred))
        
        
        # plt.plot(learningRates, stupidRes,     color='#ff0000')
        plt.plot(learningRates, perceptronRes, color='#00ff00', marker='o', ls='-', label="perceptron")
        plt.plot(learningRates, regressionRes, color='#0000ff', marker='o', ls='-', label="logistic regression")
        
        plt.legend(loc='lower right')
        plt.xscale('log')
        plt.xlabel('learning rate')
        plt.ylabel('accuracy')
        plt.title('epochs = ' + str(epochs))
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
