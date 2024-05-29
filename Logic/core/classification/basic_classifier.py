import pickle


class BasicClassifier:
    def __init__(self, model_path):
        self.model = None
        self.path = model_path

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self.path, 'rb') as f:
            self.model = pickle.load(f)

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        pass
