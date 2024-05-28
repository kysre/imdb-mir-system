import fasttext
import re
import string
import math

import unidecode
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from scipy.spatial import distance

from .fasttext_data_loader import FastTextDataLoader


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                    punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case:
        text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<br\s*/?>', '', text)
    text.strip()
    if punctuation_removal:
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
    stop_words = set(stopwords.words('english'))
    stop_words.update(stopwords_domain)
    if stopword_removal:
        new_text = ''
        lemmatizer = WordNetLemmatizer()
        tokens = pos_tag(word_tokenize(text))
        for token in tokens:
            word, tag = token
            if word not in stopwords_domain and len(word) > minimum_length:
                wntag = tag[0].lower()
                wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
                if not wntag:
                    lemma = word
                else:
                    lemma = lemmatizer.lemmatize(word, wntag)

                new_text = new_text + lemma + ' '
        text = new_text
    return text.strip()


class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None

    def train(self, texts, text_file_path='data/FastText_data.txt', should_load_data=False):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        if should_load_data:
            all_text = ''
            for text in tqdm(texts):
                all_text += text + '\n'
            with open(text_file_path, 'w', encoding='utf-8') as file:
                file.write(all_text)
                file.close()

        self.model = fasttext.train_unsupervised(text_file_path, model=self.method)
        print("Model trained successfully")

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        preprocessed_query = preprocess_text(query)
        return self.model.get_sentence_vector(preprocessed_query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # Obtain word embeddings for the words in the analogy
        embedding1 = self.model[word1]
        embedding2 = self.model[word2]
        embedding3 = self.model[word3]

        # Perform vector arithmetic
        v = embedding3 + embedding2 - embedding1

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        words = list(self.model.words.copy())

        # Exclude the input words from the possible results
        words = list(set(words).difference([word1, word2, word3]))

        # Find the word whose vector is closest to the result vector
        c_score = math.inf
        chosen_vector = None
        for word in words:
            score = distance.cosine(v, self.model[word])
            if score < c_score:
                c_score = score
                chosen_vector = word
        return chosen_vector

    def save_model(self, path='data/FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="data/FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='/Users/divar/University/term-8/information-retrieval/imdb-mir-system/Logic/data/FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)


if __name__ == "__main__":
    ft_model = FastText(method='skipgram')

    path = 'data/IMDB_crawled.json'
    ft_data_loader = FastTextDataLoader(preprocess=preprocess_text, file_path=path)

    X, y = ft_data_loader.create_train_data()
    ft_model.prepare(X, mode="train", path='data/FastText_model.bin', save=True)

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "queen"
    print(
        f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
