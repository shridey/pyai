import random

class ArtificialNeuron:

    """
    Implements the classic McCulloch-Pitts artificial neuron (1943).
    This is the simplest form of a neural unit:
    - Computes a weighted sum of inputs plus a bias.
    - Applies a step activation function (threshold at 0).
    - Outputs binary (0 or 1) predictions.

    This serves as the foundation for the Perceptron (1957) and later models.
    """

    def __init__(self, number_of_features=2, weights=None, bias=None):

        """
        Initialize the neuron with random weights and bias if not provided.

        Args:
            number_of_features (int): Number of input features (default 2).
            weights (list[float], optional): Initial weights. If None, random values in [-0.5, 0.5].
            bias (float, optional): Initial bias. If None, random value in [-0.5, 0.5].
        """

        # If weights are not supplied, initialize randomly to break symmetry.
        self.weights = weights if weights is not None else [random.uniform(-0.5, 0.5) for _ in range(number_of_features)]
        self.bias = bias if bias is not None else random.uniform(-0.5, 0.5)

    def weighted_sum(self, sample, weights, bias):

        """
        Compute the linear combination: Σ(wi * xi) + bias.

        Args:
            sample (list[float]): Input feature vector.
            weights (list[float]): Current weight vector.
            bias (float): Current bias term.

        Returns:
            float: The weighted sum.
        """

        return sum(xi * wi for xi, wi in zip(sample, weights)) + bias
    
    def activation(self, weighted_sum):

        """
        Step activation function (threshold at zero).
        Returns 1 if the weighted sum is >= 0, else 0.
        This mimics the "all-or-none" firing of a biological neuron.
        """

        return 1 if weighted_sum >= 0 else 0
    
    def predict(self, sample):

        """
        Forward pass: compute weighted sum and apply activation.

        Args:
            sample (list[float]): Input vector.

        Returns:
            int: Predicted binary class (0 or 1).
        """

        return self.activation(self.weighted_sum(sample, self.weights, self.bias))
    
    def test(self, samples, labels):

        """
        Evaluate the neuron on a set of samples and print predictions alongside true labels.
        Useful for quick sanity checks.

        Args:
            samples (list[list[float]]): List of input vectors.
            labels (list[int]): Corresponding true labels.
        """
        
        for sample, label in zip(samples, labels):
            prediction = self.predict(sample)
            print(f"Sample: {sample} | Label: {label} | Prediction: {prediction}")
