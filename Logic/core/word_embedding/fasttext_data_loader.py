import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


# from ..indexer.index_reader import Index_reader
# from ..indexer.indexes_enum import Indexes, Index_types


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """

    def __init__(self, preprocess, file_path='data/IMDB_crawled.json'):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.preprocess = preprocess
        self.file_path = file_path
        self.le = None
        self.mapping = None

    def read_data_to_df(self, should_ignore_empty_genres=True):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        with open(self.file_path, 'r') as f:
            documents = json.loads(f.read())
            f.close()
        data = []
        for doc in tqdm(documents):
            title = doc.get('title', '')
            if title is None:
                title = ''
            synopsis = doc.get('synopsis', [])
            if synopsis is None:
                synopsis = []
            summaries = doc.get('summaries', [])
            if summaries is None:
                summaries = []
            reviews = doc.get('reviews', [])
            if reviews is None:
                reviews = []
            genres = doc.get('genres', [])
            if genres is None:
                continue
            # Check for empty records
            if should_ignore_empty_genres and len(genres) == 0:
                print(f'doc_id={doc["id"]} has None genre!')
                continue
            if title == '' and len(synopsis) == len(summaries) == len(reviews) == 0:
                print(f'doc_id={doc["id"]} is None!')
                continue
            # Preprocess and add to df data
            genres = genres[0]
            data.append({
                'title': self.preprocess(title),
                'synopsis': self.preprocess(' '.join(synopsis)),
                'summaries': self.preprocess(' '.join(summaries)),
                'reviews': self.preprocess(' '.join(x[0] for x in ([['', '']] if reviews is None or len(reviews) == 0 else reviews))),
                'genres': self.preprocess(genres),
            })
        return pd.DataFrame(data)

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        self.le = LabelEncoder()
        df['genres'] = self.le.fit_transform(df['genres'])
        self.mapping = dict(zip(range(len(self.le.classes_)), self.le.classes_))
        df['text'] = df['synopsis'] + ' ' + df['summaries'] + ' ' + df['reviews'] + ' ' + df['title']
        x = np.array(df['text'])
        y = np.array(df['genres'])
        return x, y
