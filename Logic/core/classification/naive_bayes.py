import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__(
            model_path='/Users/divar/University/term-8/information-retrieval/imdb-mir-system/'
                       'Logic/data/classification/naivebayes.pkl'
        )
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape
        self.prior = []
        for i in range(self.num_classes):
            self.prior.append(sum(y == i) / self.number_of_samples)
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        for i in range(self.num_classes):
            x_class = x[np.argwhere(y == i)]
            num_words = np.sum(x_class, axis=0)
            self.feature_probabilities[i, :] = (num_words + self.alpha) / (np.sum(x_class) + self.number_of_features)
        return self

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
        n, _ = x.shape
        log_predict_probs = np.zeros((n, self.num_classes))
        for i in range(self.num_classes):
            log_predict_probs[:, i] += np.log(self.prior[i])
            class_feature_prob = np.log(self.feature_probabilities[i, :].reshape(self.number_of_features, 1))
            log_predict_probs[:, i] += (x @ class_feature_prob).squeeze()
        return np.argmax(log_predict_probs, axis=1)

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

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        x = self.cv.transform(sentences).toarray()
        pred = self.predict(x)
        return sum(pred) / len(pred)


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the reviews using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn 's CountVectorizer to find the embeddings.
    """
    # My laptop will run out of memory when trying to train the model for all reviews
    loader = ReviewLoader()
    loader.load_data()
    reviews = loader.df['review'].values[:20000]
    sentiments = loader.df['sentiment'].values[:20000]

    cv = CountVectorizer()
    cv.fit(reviews)

    x_train, x_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)
    x_train = cv.transform(x_train).toarray()
    x_test = cv.transform(x_test).toarray()

    classifier = NaiveBayes(cv)
    classifier.fit(x_train, y_train)
    result = classifier.prediction_report(x_test, y_test)
    print(result)
