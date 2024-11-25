import os
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.means = None
        self.vars = None
        self.priors = None

    # Train the Gaussian Naive Bayes model
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        self.means = np.zeros((len(self.classes), n_features))
        self.vars = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))

        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.means[idx, :] = X_cls.mean(axis=0)
            self.vars[idx, :] = X_cls.var(axis=0)
            self.priors[idx] = X_cls.shape[0] / float(n_samples)

    # Predict the labels for the input data
    def predict(self, X):
        posteriors = np.apply_along_axis(self._calculate_posterior, axis=1, arr=X)
        return self.classes[np.argmax(posteriors, axis=1)]

    # Calculate the posterior probability for a given input
    def _calculate_posterior(self, x):
        posteriors = []
        for idx, cls in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.vars[idx]))
            likelihood -= 0.5 * np.sum(((x - self.means[idx]) ** 2) / (self.vars[idx]))
            posteriors.append(prior + likelihood)
        return posteriors

    # Save the trained model to a file
    def save(self, filename):
        joblib.dump(self, filename)
        print(f"Custom Naive Bayes model saved to {filename}")

    @staticmethod
    def load(filename):
        model = joblib.load(filename)
        print(f"Custom Naive Bayes model loaded from {filename}")
        return model

# Evaluate the model and display accuracy, confusion matrix, and classification report
def evaluate(predictions, y_test, method="Custom"):
    accuracy = accuracy_score(y_test, predictions)
    print(f"{method} Naive Bayes Accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Load features
    X_train = np.load("train_features.npy")
    y_train = np.load("train_labels.npy")
    X_test = np.load("test_features.npy")
    y_test = np.load("test_labels.npy")

    # Prompt user for model name
    custom_model_path = "models/custom_naive_bayes.pkl"
    sklearn_model_path = "models/sklearn_naive_bayes.pkl"

    # Custom Naive Bayes
    print("Custom Naive Bayes:")
    if os.path.exists(custom_model_path):
        load_existing = input(f"Custom model '{custom_model_path}' exists. Load it? (y/n): ").strip().lower()
        if load_existing == 'y':
            gnb_custom = GaussianNaiveBayes.load(custom_model_path)
        else:
            gnb_custom = GaussianNaiveBayes()
            gnb_custom.fit(X_train, y_train)
            gnb_custom.save(custom_model_path)
    else:
        gnb_custom = GaussianNaiveBayes()
        gnb_custom.fit(X_train, y_train)
        gnb_custom.save(custom_model_path)

    predictions_custom = gnb_custom.predict(X_test)
    print("\nCustom Naive Bayes Evaluation:")
    evaluate(predictions_custom, y_test, method="Custom")

    # Scikit-learn Naive Bayes
    print("\nScikit-learn Naive Bayes:")
    if os.path.exists(sklearn_model_path):
        load_existing = input(f"Scikit-learn model '{sklearn_model_path}' exists. Load it? (y/n): ").strip().lower()
        if load_existing == 'y':
            gnb_sklearn = joblib.load(sklearn_model_path)
            print(f"Scikit-learn Naive Bayes model loaded from {sklearn_model_path}")
        else:
            gnb_sklearn = GaussianNB()
            gnb_sklearn.fit(X_train, y_train)
            joblib.dump(gnb_sklearn, sklearn_model_path)
            print(f"Scikit-learn model saved to {sklearn_model_path}")
    else:
        gnb_sklearn = GaussianNB()
        gnb_sklearn.fit(X_train, y_train)
        joblib.dump(gnb_sklearn, sklearn_model_path)
        print(f"Scikit-learn model saved to {sklearn_model_path}")

    predictions_sklearn = gnb_sklearn.predict(X_test)
    print("\nScikit-learn Naive Bayes Evaluation:")
    evaluate(predictions_sklearn, y_test, method="Scikit-learn")
