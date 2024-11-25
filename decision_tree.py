import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import time

# Custom Decision Tree Implementation
class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    # Build the decision tree using recursive splitting.
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    # Recursively split the data to build the tree.
    def _build_tree(self, X, y, depth):
        print(f"Building tree at depth {depth}, number of samples: {len(y)}")

        # Base cases
        if len(np.unique(y)) == 1:  # If all samples belong to one class
            return {"label": y[0]}
        if depth >= self.max_depth or len(y) == 0:
            return {"label": np.bincount(y).argmax()}  # Majority class

        # Find the best split
        feature_idx, threshold = self._find_best_split(X, y)
        if feature_idx is None:
            return {"label": np.bincount(y).argmax()}  # Majority class

        # Split the data
        left_idx = X[:, feature_idx] <= threshold
        right_idx = ~left_idx
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            "feature": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }
    # Find the best feature and threshold to split on using Gini impurity
    def _find_best_split(self, X, y):
        print("Finding the best split...")
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            print(f"Evaluating feature {feature_idx} with {len(thresholds)} thresholds...")
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gini = self._gini_index(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        print(f"Best feature: {best_feature}, Best threshold: {best_threshold}")
        return best_feature, best_threshold

    # Calculate the Gini impurity index for a split.
    def _gini_index(self, left_labels, right_labels):
        def gini(labels):
            probs = np.bincount(labels) / len(labels)
            return 1 - np.sum(probs ** 2)

        n_left = len(left_labels)
        n_right = len(right_labels)
        total = n_left + n_right

        return (n_left / total) * gini(left_labels) + (n_right / total) * gini(right_labels)

    # Predict labels for the input data using the trained tree.
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    # Traverse the tree to predict the label for a single sample.
    def _traverse_tree(self, x, node):
        if "label" in node:
            return node["label"]
        feature = node["feature"]
        threshold = node["threshold"]

        if x[feature] <= threshold:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])

    # Save the trained tree to a file.
    def save(self, filename):
        joblib.dump(self.tree, filename)
        print(f"Tree saved to {filename}")

    # Load a saved tree from a file.
    def load(self, filename):
        self.tree = joblib.load(filename)
        print(f"Tree loaded from {filename}")


# Evaluate Model
def evaluate(predictions, y_test, method="Custom"):
    accuracy = accuracy_score(y_test, predictions)
    print(f"{method} Decision Tree Accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    # Load features
    X_train = np.load("train_features.npy")
    y_train = np.load("train_labels.npy")
    X_test = np.load("test_features.npy")
    y_test = np.load("test_labels.npy")

    # Prompt user for desired depth
    while True:
        try:
            max_depth = int(input("Enter the desired tree depth (e.g., 10, 25, 50): ").strip())
            break
        except ValueError:
            print("Invalid input. Please enter an integer value for depth.")

    # File paths based on depth
    custom_model_path = f"models/custom_tree_depth_{max_depth}.pkl"
    sklearn_model_path = f"models/sklearn_tree_depth_{max_depth}.pkl"

    # Custom Decision Tree
    print(f"\nCustom Decision Tree with max_depth={max_depth}:")
    dt_custom = DecisionTree(max_depth=max_depth)

    if os.path.exists(custom_model_path):
        load_existing = input(f"Custom model '{custom_model_path}' exists. Load it? (y/n): ").strip().lower()
        if load_existing == 'y':
            dt_custom.load(custom_model_path)
            print("\nLoaded saved Custom Decision Tree.")
        else:
            start_time = time.time()
            dt_custom.fit(X_train, y_train)
            end_time = time.time()
            print(f"Custom Decision Tree Training Time: {end_time - start_time:.2f} seconds")
            dt_custom.save(custom_model_path)
    else:
        start_time = time.time()
        dt_custom.fit(X_train, y_train)
        end_time = time.time()
        print(f"Custom Decision Tree Training Time: {end_time - start_time:.2f} seconds")
        dt_custom.save(custom_model_path)

    # Evaluate Custom Decision Tree
    predictions_custom = dt_custom.predict(X_test)
    evaluate(predictions_custom, y_test, method="Custom")

    # Scikit-learn Decision Tree
    print(f"\nScikit-learn Decision Tree with max_depth={max_depth}:")
    if os.path.exists(sklearn_model_path):
        load_existing = input(f"Scikit-learn model '{sklearn_model_path}' exists. Load it? (y/n): ").strip().lower()
        if load_existing == 'y':
            dt_sklearn = joblib.load(sklearn_model_path)
            print("\nLoaded saved Scikit-learn Decision Tree.")
        else:
            dt_sklearn = DecisionTreeClassifier(max_depth=max_depth)
            start_time = time.time()
            dt_sklearn.fit(X_train, y_train)
            end_time = time.time()
            print(f"Scikit-learn Decision Tree Training Time: {end_time - start_time:.2f} seconds")
            joblib.dump(dt_sklearn, sklearn_model_path)
            print(f"Scikit-learn model saved to {sklearn_model_path}")
    else:
        dt_sklearn = DecisionTreeClassifier(max_depth=max_depth)
        start_time = time.time()
        dt_sklearn.fit(X_train, y_train)
        end_time = time.time()
        print(f"Scikit-learn Decision Tree Training Time: {end_time - start_time:.2f} seconds")
        joblib.dump(dt_sklearn, sklearn_model_path)
        print(f"Scikit-learn model saved to {sklearn_model_path}")

    # Evaluate Scikit-learn Decision Tree
    predictions_sklearn = dt_sklearn.predict(X_test)
    evaluate(predictions_sklearn, y_test, method="Scikit-learn")