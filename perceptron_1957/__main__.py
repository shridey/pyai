from perceptron_1957.perceptron import Perceptron

'''
Demonstration of Perceptron predicting AND Logic Gate
using learned weights
'''

samples = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels_and = [0, 0, 0, 1]

perceptron = Perceptron(2, 0.1, 10)

print("Testing the AND Logic Gate Prediction Accuracy of Perceptron before training:")
perceptron.test(samples, labels_and)

print()

print("Begin training")
print()
perceptron.train(samples, labels_and)

print()

print("Testing the AND Logic Gate Prediction Accuracy of Perceptron after training:")
perceptron.test(samples, labels_and)
