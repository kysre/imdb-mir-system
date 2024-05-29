import numpy as np
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from tqdm import tqdm
import scipy.spatial
from collections import Counter

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class KnnClassifierData:
    def __init__(self):
        self.k = None
        self.pca = None
        self.X_train = None
        self.y_train = None


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__(
            model_path='/Users/divar/University/term-8/information-retrieval/imdb-mir-system/'
                       'Logic/data/classification/knn.pkl'
        )
        self.model = KnnClassifierData()
        self.model.k = n_neighbors
        self.model.pca = PCA(n_components=20)
        self.model.X_train = None
        self.model.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.model.pca.fit(np.array(list(x)))
        self.model.X_train = self.model.pca.transform(np.array(list(x)))
        self.model.y_train = y

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
        x_reduced = self.model.pca.transform(np.array(list(x)))
        predictions = []
        for i in tqdm(range(len(x_reduced))):
            d = []
            votes = []
            for j in range(len(self.model.X_train)):
                dist = scipy.spatial.distance.euclidean(self.model.X_train[j], x_reduced[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.model.k]
            for d, j in d:
                votes.append(y_train[j])
            ans = Counter(votes).most_common(1)[0][0]
            predictions.append(ans)
        return predictions

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


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader()
    loader.load_data()
    classifier = KnnClassifier(n_neighbors=3)
    x_train, x_test, y_train, y_test = loader.split_data()
    classifier.fit(x_train, y_train)
    classifier.save()
    result = classifier.prediction_report(x_test, y_test)
    print(result)
