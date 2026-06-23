import random

class ArtificialNeuron:
    def __init__(self, number_of_features=2, weights=None, bias=None):
        self.weights = weights if weights is not None else [random.uniform(-0.5, 0.5) for _ in range(number_of_features)]
        self.bias = bias if bias is not None else random.uniform(-0.5, 0.5)

    def weighted_sum(self, sample, weights, bias):
        return sum(xi * wi for xi, wi in zip(sample, weights)) + bias
    
    def activation(self, weighted_sum):
        return 1 if weighted_sum >= 0 else 0
    
    def predict(self, sample):
        return self.activation(self.weighted_sum(sample, self.weights, self.bias))
    
    def test(self, samples, labels):
        for sample, label in zip(samples, labels):
            prediction = self.predict(sample)
            print(f"Sample: {sample} | Label: {label} | Prediction: {prediction}")