import random

class Perceptron:
    def __init__(self, number_of_features=2, learning_rate=0.1, max_epochs=10, weights=None, bias=None):
        self.number_of_features = number_of_features
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = weights if weights is not None else [random.uniform(-0.5, 0.5) for _ in range(number_of_features)]
        self.bias = bias if bias is not None else random.uniform(-0.5, 0.5)

    def weighted_sum(self, sample, weights, bias):
        return sum(xi * wi for xi, wi in zip(sample, weights)) + bias
    
    def activation(self, weighted_sum):
        return 1 if weighted_sum >= 0 else 0
    
    def predict(self, sample):
        return self.activation(self.weighted_sum(sample, self.weights,self.bias))
    
    def update(self, sample, error):
        self.weights = [wi + self.learning_rate * error * xi for wi, xi in zip(self.weights, sample)]
        self.bias += self.learning_rate * error

    def train(self, samples, labels, verbose=True):
        if verbose:
            print("Pre-training State:")
            print(f"Weights: {self.weights}")
            print(f"Bias: {self.bias}")
            print()
            
        for epoch in range(self.max_epochs):
            any_error = False
            for sample, label in zip(samples, labels):
                prediction = self.predict(sample)

                if prediction != label:
                    any_error = True
                    error = label - prediction
                    self.update(sample, error)
                
                if verbose:
                    print(f"Epoch: {epoch+1} | Sample: {sample} | Label: {label} | Prediction: {prediction} | Error: {label - prediction} | Updated Weights: {self.weights} | Updated Bias: {self.bias}")

            if verbose:
                print()

            if not any_error:
                if verbose:
                    print(f"Converging at epoch {epoch+1}")
                    print()
                    print("Post-training State:")
                    print(f"Weights: {self.weights}")
                    print(f"Bias: {self.bias}")

                return dict({"trained_weights": self.weights, "trained_bias": self.bias})

        if verbose:
            print("Post-training State:")
            print("Epochs exhausted but predictions are still inaccurate")
            print(f"Weights: {self.weights}")
            print(f"Bias: {self.bias}")

        return dict({"trained_weights": self.weights, "trained_bias": self.bias})

    def test(self, samples, labels):
        for sample, label in zip(samples, labels):
            prediction = self.predict(sample)
            print(f"Sample: {sample} | Label: {label} | Prediction: {prediction}")
