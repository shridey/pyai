from perceptron import Perceptron

samples = [[0, 0], [0, 1], [1, 0], [1, 1]]

labels_and = [0, 0, 0, 1]
labels_nand = [1, 1, 1, 0]
labels_or = [0, 1, 1, 1]
labels_nor = [1, 0, 0, 0]
labels_xor = [0, 1, 1, 0]
labels_xnor = [1, 0, 0, 1]

perceptron = Perceptron()

print("Testing accuracy before training:")
perceptron.test(samples, labels_and)

print()

perceptron.train(samples, labels_and)

print()

print("Testing accuracy after training:")
perceptron.test(samples, labels_and)
