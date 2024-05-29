import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from word_embedding.fasttext_model import FastText, preprocess_text


class ReviewLoader:
    def __init__(
            self,
            file_path: str = '/Users/divar/University/term-8/information-retrieval/imdb-mir-system/Logic/data/classification.pkl',
            comments_path: str = '/Users/divar/University/term-8/information-retrieval/imdb-mir-system/Logic/data/comments_training.csv'
    ):
        self.file_path = file_path
        self.comments_path = comments_path
        self.df = None
        self.fasttext_model = FastText()
        self.fasttext_model.prepare(dataset=[], mode='load', save=False)
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def save_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        self.df = pd.read_csv(self.comments_path)
        self.df['review'] = self.df['review'].apply(preprocess_text)
        self.df['review_embedding'] = self.df['review'].apply(self.fasttext_model.get_query_embedding)
        mymap = {'positive': 1, 'negative': 0}
        self.df['sentiment'] = self.df['sentiment'].apply(lambda s: mymap.get(s) if s in mymap else s)
        self.df['review_embedding'] = self.df['review_embedding'].apply(list)
        self.df.to_pickle(self.file_path)

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        self.df = pd.read_pickle(self.file_path)

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        pass

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        threshold = int(test_data_ratio * self.df.shape[0])
        train_df, test_df = train_test_split(self.df, test_size=threshold, random_state=42)
        return (
            train_df['review_embedding'].values, test_df['review_embedding'].values,
            train_df['sentiment'].values, test_df['sentiment'].values
        )


if __name__ == '__main__':
    review_loader = ReviewLoader()
    review_loader.save_data()
