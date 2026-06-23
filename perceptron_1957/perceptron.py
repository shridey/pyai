from artificial_neuron_1943.artificial_neuron import ArtificialNeuron

class Perceptron(ArtificialNeuron):
    def __init__(self, number_of_features=2, learning_rate=0.1, max_epochs=10, weights=None, bias=None):
        super().__init__(number_of_features, weights, bias)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

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

                return {"trained_weights": self.weights, "trained_bias": self.bias}

        if verbose:
            print("Post-training State:")
            print("Epochs exhausted but predictions are still inaccurate")
            print(f"Weights: {self.weights}")
            print(f"Bias: {self.bias}")

        return {"trained_weights": self.weights, "trained_bias": self.bias}