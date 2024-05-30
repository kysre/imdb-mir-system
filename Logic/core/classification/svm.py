import pickle

import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__(
            model_path='/Users/divar/University/term-8/information-retrieval/imdb-mir-system/'
                       'Logic/data/classification/svm.pkl'
        )
        self.model = SVC(kernel='linear', random_state=42)

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        return self.model.predict(x)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)


# F1 accuracy : 78%
# F1 acquired : 86%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader()
    loader.load_data()
    x_train, x_test, y_train, y_test = loader.split_data()
    classifier = SVMClassifier()
    classifier.fit(x_train, y_train)
    classifier.save()
    result = classifier.prediction_report(x_test, y_test)
    print(result)
