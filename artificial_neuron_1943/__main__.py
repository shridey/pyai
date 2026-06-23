from artificial_neuron_1943.artificial_neuron import ArtificialNeuron

'''
Demonstration of Artificial Neuron predicting AND Logic Gate
using hardcoded weights
'''

samples = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels_and = [0, 0, 0, 1]

'''
Calculating weights for AND Gate:

x1 | x2 | x1 AND x2
0  | 0  | 0
0  | 1  | 0
1  | 0  | 0
1  | 1  | 1

Artificial Neuron Activation Function: w1.x1 + w2.x2 + b >= 0
It means that the Artificial Neuron will output 1 only when
weighted sum will be greater than or equal to 0.

Sample 1: x1 = 0, x2 = 0 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.0 + w2.0 + b < 0
b < 0 ... (Condition 1)

Sample 2: x1 = 0, x2 = 1 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.(0) + w2.(1) + b < 0
w2 + b < 0
w2 < -b ... (Condition 2)

Sample 3: x1 = 1, x2 = 0 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.(1) + w2.(0) + b < 0
w1 + b < 0
w1 < -b ... (Condition 3)

Sample 4: x1 = 1, x2 = 1 | Target = 1
w1.x1 + w2.x2 + b >= 0
w1.(1) + w2.(1) + b >= 0
w1 + w2 + b >= 0
w1 + w2 >= -b ... (Condition 4)

For satisfying all the four conditions of inequalities,
Let b = -1.5, w1 = 1, w2 = 1
'''
artificial_neuron = ArtificialNeuron(2, [1, 1], -1.5)

print("Prediction of AND GATE by Artificial Neuron:")
artificial_neuron.test(samples, labels_and)