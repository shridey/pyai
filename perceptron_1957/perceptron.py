from artificial_neuron_1943.artificial_neuron import ArtificialNeuron

class Perceptron(ArtificialNeuron):

    """
    Extends the ArtificialNeuron with the Perceptron learning rule (Rosenblatt, 1957).
    Key additions:
    - Learning rate and maximum epochs.
    - Iterative weight update based on prediction errors.
    - Training method that converges when all samples are correctly classified.

    This is a single-layer, binary classifier that can learn linearly separable patterns.
    """

    def __init__(self, number_of_features=2, learning_rate=0.1, max_epochs=10, weights=None, bias=None):
        
        """
        Initialize the Perceptron with training hyperparameters.

        Args:
            number_of_features (int): Number of input features.
            learning_rate (float): Step size for weight updates (controls convergence speed).
            max_epochs (int): Maximum number of full passes over the training data.
            weights (list[float], optional): Initial weights (inherited from parent).
            bias (float, optional): Initial bias (inherited from parent).
        """
        
        # Call parent constructor to set up weights and bias
        super().__init__(number_of_features, weights, bias)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def update(self, sample, error):

        """
        Apply the Perceptron update rule:
            Δwi = learning_rate * error * xi
            Δbias = learning_rate * error

        Only updates when error != 0 (i.e., misclassification).

        Args:
            sample (list[float]): The input vector that caused the error.
            error (int): The difference (true_label - predicted_label), which is either -1, 0, or +1.
        """

        self.weights = [wi + self.learning_rate * error * xi for wi, xi in zip(self.weights, sample)]
        self.bias += self.learning_rate * error

    def train(self, samples, labels, verbose=True):

        """
        Train the perceptron using the delta rule.

        The algorithm:
        1. For each epoch, iterate over all training samples.
        2. If a sample is misclassified, update weights and bias.
        3. Stop early if all samples are correctly classified (convergence).
        4. Otherwise, continue until max_epochs is reached.

        Args:
            samples (list[list[float]]): Training input vectors.
            labels (list[int]): True binary labels (0 or 1).
            verbose (bool): If True, print detailed training progress.

        Returns:
            dict: Contains the final trained weights and bias.
        """

        if verbose:
            print("Pre-training State:")
            print(f"Weights: {self.weights}")
            print(f"Bias: {self.bias}")
            print()

        for epoch in range(self.max_epochs):
            any_error = False # Flag to track if any sample was misclassified this epoch
            for sample, label in zip(samples, labels):
                prediction = self.predict(sample)

                # Check for misclassification
                if prediction != label:
                    any_error = True
                    error = label - prediction # error ∈ {-1, 1}
                    self.update(sample, error)
                
                if verbose:
                    # Detailed log for each sample: shows the dynamic learning process
                    print(f"Epoch: {epoch+1} | Sample: {sample} | Label: {label} | Prediction: {prediction} | Error: {label - prediction} | Updated Weights: {self.weights} | Updated Bias: {self.bias}")

            if verbose:
                print() # Blank line between epochs

            # If no errors occurred in this epoch, the model has converged
            if not any_error:
                if verbose:
                    print(f"Converging at epoch {epoch+1}")
                    print()
                    print("Post-training State:")
                    print(f"Weights: {self.weights}")
                    print(f"Bias: {self.bias}")

                return {"trained_weights": self.weights, "trained_bias": self.bias}

        # If we exit the loop without convergence, warn the user
        if verbose:
            print("Post-training State:")
            print("Epochs exhausted but predictions are still inaccurate")
            print(f"Weights: {self.weights}")
            print(f"Bias: {self.bias}")

        return {"trained_weights": self.weights, "trained_bias": self.bias}
