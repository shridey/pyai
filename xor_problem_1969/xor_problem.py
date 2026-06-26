from perceptron_1957.perceptron import Perceptron

# Demonstration of how a single Perceptron could not learn XOR Logic Gate

samples = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels_xor = [0, 1, 1, 0]

'''
Let's understand why a single Perceptron could not learn XOR Logic Gate

Calculating weights for XOR Logic Gate

x1 | x2 | x1 XOR x2
0  | 0  | 0
0  | 1  | 1
1  | 0  | 1
1  | 1  | 0

Perceptron Activation Function: w1.x1 + w2.x2 + b >= 0
It means that the Perceptron will output 1 only when
weighted sum will be greater than or equal to 0.

Sample 1: x1 = 0, x2 = 0 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.0 + w2.0 + b < 0
b < 0 ... (Condition 1)

Sample 2: x1 = 0, x2 = 1 | Target = 1
w1.x1 + w2.x2 + b >= 0
w1.(0) + w2.(1) + b >= 0
w2 + b >= 0
w2 >= -b ... (Condition 2)

Sample 3: x1 = 1, x2 = 0 | Target = 1
w1.x1 + w2.x2 + b >= 0
w1.(1) + w2.(0) + b >= 0
w1 + b >= 0
w1 >= -b ... (Condition 3)

Sample 4: x1 = 1, x2 = 1 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.(1) + w2.(1) + b < 0
w1 + w2 + b < 0
w1 + w2 < -b ... (Condition 4)

For satisfying all the four conditions of inequalities,
there are no valid values.

Hence, a single Perceptron will never find the weights
which could accurately predict the XOR Logic Gate
as XOR is not linearly seperable
'''

perceptron = Perceptron(2, 0.1, 10)

print("Testing the XOR Logic Gate Prediction Accuracy of Perceptron before training:")
perceptron.test(samples, labels_xor)

print()

print("Begin training")
print()
perceptron.train(samples, labels_xor)

print()

print("Testing the XOR Logic Gate Prediction Accuracy of Perceptron after training:")
perceptron.test(samples, labels_xor)