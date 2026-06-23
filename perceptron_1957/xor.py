from perceptron_1957.perceptron import Perceptron

'''
Demonstration of Multiple Perceptrons predicting XOR Logic Gate
using hardcoded weights
'''

samples = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels_xor = [0, 1, 1, 0]

'''
Calculating weights for XOR Logic Gate

x1 | x2 | x1 XOR x2
0  | 0  | 0
0  | 1  | 1
1  | 0  | 1
1  | 1  | 0

A single Perceptron could not learn XOR Logic Gate, here's why:

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

Hence, we need multiple Perceptrons to break this problem into multiple steps.
The idea is to train first Perceptron p1 to learn:
x1 | x2 | NOT x1 AND x2
0  | 0  | 0
0  | 1  | 1
1  | 0  | 0
1  | 1  | 0

and train second Perceptron p2 to learn:
x1 | x2 | x1 AND NOT x2
0  | 0  | 0
0  | 1  | 0
1  | 0  | 1
1  | 1  | 0

then train third Percetron p3 to learn:
p1 | p2 | p1 OR p2
0  | 0  | 0
1  | 0  | 1
0  | 1  | 1
0  | 0  | 0

Lets calculate weights for all the three Perceptrons

For Perceptron p1 to learn NOT x1 AND x2
x1 | x2 | NOT x1 AND x2
0  | 0  | 0
0  | 1  | 1
1  | 0  | 0
1  | 1  | 0

Sample 1: x1 = 0, x2 = 0 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.0 + w2.0 + b < 0
b < 0 ... (Condition 1)

Sample 2: x1 = 0, x2 = 1 | Target = 1
w1.x1 + w2.x2 + b >= 0
w1.(0) + w2.(1) + b >= 0
w2 + b >= 0
w2 >= -b ... (Condition 2)

Sample 3: x1 = 1, x2 = 0 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.(1) + w2.(0) + b < 0
w1 + b < 0
w1 < -b ... (Condition 3)

Sample 4: x1 = 1, x2 = 1 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.(1) + w2.(1) + b < 0
w1 + w2 + b < 0
w1 + w2 < -b ... (Condition 4)

For satisfying all the four conditions of inequalities,
Let b = -0.5, w1 = -1, w2 = 1

For Perceptron p2 to learn x1 AND NOT x2
x1 | x2 | x1 AND NOT x2
0  | 0  | 0
0  | 1  | 0
1  | 0  | 1
1  | 1  | 0

Sample 1: x1 = 0, x2 = 0 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.0 + w2.0 + b < 0
b < 0 ... (Condition 1)

Sample 2: x1 = 0, x2 = 1 | Target = 0
w1.x1 + w2.x2 + b < 0
w1.(0) + w2.(1) + b < 0
w2 + b < 0
w2 < -b ... (Condition 2)

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
Let b = -0.5, w1 = 1, w2 = -1

For Perceptron p3 to learn p1 OR p2
p1 | p2 | p1 OR p2
0  | 0  | 0
1  | 0  | 1
0  | 1  | 1
0  | 0  | 0

Sample 1: p1 = 0, p2 = 0 | Target = 0
w1.p1 + w2.p2 + b < 0
w1.0 + w2.0 + b < 0
b < 0 ... (Condition 1)

Sample 2: p1 = 0, p2 = 1 | Target = 1
w1.p1 + w2.p2 + b >= 0
w1.(0) + w2.(1) + b >= 0
w2 + b >= 0
w2 >= -b ... (Condition 2)

Sample 3: p1 = 1, p2 = 0 | Target = 1
w1.p1 + w2.p2 + b >= 0
w1.(1) + w2.(0) + b >= 0
w1 + b >= 0
w1 >= -b ... (Condition 3)

Sample 4: p1 = 1, p2 = 1 | Target = 0
w1.p1 + w2.p2 + b < 0
w1.(1) + w2.(1) + b < 0
w1 + w2 + b < 0
w1 + w2 < -b ... (Condition 4)

For satisfying all the four conditions of inequalities,
Let b = -0.5, w1 = 1, w2 = 1
'''

p1 = Perceptron(weights=[-1, 1], bias=-0.5)
p2 = Perceptron(weights=[1, -1], bias=-0.5)

p1_output_sample_1 = p1.predict(samples[0])
p1_output_sample_2 = p1.predict(samples[1])
p1_output_sample_3 = p1.predict(samples[2])
p1_output_sample_4 = p1.predict(samples[3])

p2_output_sample_1 = p2.predict(samples[0])
p2_output_sample_2 = p2.predict(samples[1])
p2_output_sample_3 = p2.predict(samples[2])
p2_output_sample_4 = p2.predict(samples[3])

p3 = Perceptron(weights=[1, 1], bias=-0.5)

p3_output_sample_1 = p3.predict([p1_output_sample_1, p2_output_sample_1])
p3_output_sample_2 = p3.predict([p1_output_sample_2, p2_output_sample_2])
p3_output_sample_3 = p3.predict([p1_output_sample_3, p2_output_sample_3])
p3_output_sample_4 = p3.predict([p1_output_sample_4, p2_output_sample_4])

print("XOR Logic Gate Prediction by Multiple Perceptrons:")
print(f"Sample: {samples[0]} | Label: {labels_xor[0]} | Prediction: {p3_output_sample_1}")
print(f"Sample: {samples[1]} | Label: {labels_xor[1]} | Prediction: {p3_output_sample_2}")
print(f"Sample: {samples[2]} | Label: {labels_xor[2]} | Prediction: {p3_output_sample_3}")
print(f"Sample: {samples[3]} | Label: {labels_xor[3]} | Prediction: {p3_output_sample_4}")
