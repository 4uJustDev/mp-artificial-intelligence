import numpy as np
import json


class Perceptron:
    def __init__(self, input_size, num_classes, learning_rate=0.1):
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # Initialize weights with better scaling
        self.weights = np.random.randn(num_classes, input_size) * np.sqrt(
            2.0 / input_size
        )
        self.bias = np.zeros(num_classes)
        self.training_history = []

    def load_weights(self, filename):
        """Загружает веса и смещения из файла"""
        with open(filename, "r") as f:
            weights_dict = json.load(f)
        self.weights = np.array(weights_dict["weights"])
        self.bias = np.array(weights_dict["bias"])
        self.input_size = weights_dict["input_size"]
        self.num_classes = weights_dict["num_classes"]

    def _one_hot_encode(self, y):
        """Convert class labels to one-hot encoding"""
        one_hot = np.zeros((len(y), self.num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def predict(self, x, return_details=False):
        # Compute raw outputs
        raw_output = np.dot(self.weights, x) + self.bias
        # Apply softmax activation for multi-class classification
        exp_output = np.exp(
            raw_output - np.max(raw_output)
        )  # Subtract max for numerical stability
        activated_output = exp_output / np.sum(exp_output)

        if return_details:
            return {
                "predicted_class": np.argmax(activated_output),
                "raw_output": raw_output,
                "activated_output": activated_output,
                "probabilities": activated_output,
            }
        return np.argmax(activated_output)

    def train(self, X, y, max_epochs=1000, error_threshold=0.001):
        """Обучение персептрона с сохранением истории"""
        self.training_history = []
        y_one_hot = self._one_hot_encode(y)
        best_error = float("inf")
        patience = 10  # Early stopping patience
        no_improvement = 0

        for epoch in range(max_epochs):
            total_error = 0
            correct_predictions = 0

            # Shuffle the training data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            for i in range(len(X_shuffled)):
                # Forward pass
                raw_output = np.dot(self.weights, X_shuffled[i]) + self.bias
                exp_output = np.exp(raw_output - np.max(raw_output))
                activated_output = exp_output / np.sum(exp_output)

                # Calculate error
                error = y_shuffled[i] - activated_output
                total_error += np.sum(np.abs(error))

                # Update weights and bias
                self.weights += self.learning_rate * np.outer(error, X_shuffled[i])
                self.bias += self.learning_rate * error

                # Count correct predictions
                if np.argmax(activated_output) == np.argmax(y_shuffled[i]):
                    correct_predictions += 1

            # Calculate accuracy
            accuracy = correct_predictions / len(X)

            # Save training history
            self.training_history.append(
                {"epoch": epoch + 1, "total_error": total_error, "accuracy": accuracy}
            )

            print(
                f"Эпоха {epoch + 1}, Ошибка: {total_error:.4f}, Точность: {accuracy:.4f}"
            )

            # Early stopping
            if total_error < best_error:
                best_error = total_error
                no_improvement = 0
            else:
                no_improvement += 1

            if total_error < error_threshold or no_improvement >= patience:
                print(f"Обучение остановлено на эпохе {epoch + 1}")
                break

        return self.training_history

    def get_network_state(self):
        """Возвращает текущее состояние сети"""
        return {
            "weights": self.weights,
            "bias": self.bias,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
        }
