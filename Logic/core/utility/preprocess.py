import json
import string
import re
from typing import List

import nltk
import unidecode
from tqdm import tqdm


class Preprocessor:
    @classmethod
    def get_stopwords(cls, use_local_stopwords=False) -> List[str]:
        if use_local_stopwords:
            with open('data/stopwords.txt', 'r') as f:
                stopwords = f.read()
            return stopwords.split()
        return nltk.corpus.stopwords.words('english')

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        self.stopwords = Preprocessor.get_stopwords()

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
        str
            The preprocessed documents.
        """
        preprocessed_documents = []
        for document in tqdm(self.documents):
            if isinstance(document, str):
                self.documents = [self.normalize(document)]
                preprocessed_documents.extend(self.documents.copy())
                break
            else:
                none_count = 0
                for key in document:
                    if document[key] is None:
                        none_count += 1
                        document[key] = self.get_default(key)
                    elif isinstance(document[key], list):
                        preprocessed_doc = []
                        if len(document[key]) != 0:
                            if isinstance(document[key][0], list):
                                # reviews
                                for review, score in document[key]:
                                    preprocessed_doc.append([self.normalize(review), score])
                            else:
                                preprocessed_doc = [self.normalize(text) for text in document[key]]
                        document[key] = preprocessed_doc
                    else:
                        document[key] = self.normalize(document[key])
            if none_count < len(document.keys()) - 1:
                preprocessed_documents.append(document)
        return preprocessed_documents

    def get_default(self, key: str):
        """
        returns a default value for None fields
        """
        default_dict = {
            'title': '',  # str
            'first_page_summary': '',  # str
            'release_year': '',  # str
            'mpaa': '',  # str
            'budget': '',  # str
            'gross_worldwide': '',  # str
            'rating': '',  # str
            'directors': [],  # List[str]
            'writers': [],  # List[str]
            'stars': [],  # List[str]
            'related_links': [],  # List[str]
            'genres': [],  # List[str]
            'languages': [],  # List[str]
            'countries_of_origin': [],  # List[str]
            'summaries': [],  # List[str]
            'synopsis': [],  # List[str]
            'reviews': [],  # List[List[str]]
        }
        return default_dict.get(key, None)

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """

        text = unidecode.unidecode(text)
        text = text.lower()
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text.strip()

        text = self.remove_links(text)

        text = self.remove_punctuations(text)

        non_stopwords = self.remove_stopwords(text)
        text = ' '.join(non_stopwords)

        tokens = self.tokenize(text)

        stemmer = nltk.PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        return ' '.join(stemmed_tokens)

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        tokens = nltk.word_tokenize(text)
        return tokens

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        words = text.split()
        non_stopwords = []
        for word in words:
            if word not in self.stopwords:
                non_stopwords.append(word)
        return non_stopwords


if __name__ == "__main__":
    with open('data/IMDB_crawled.json', 'r') as f:
        data = json.loads(f.read())
        f.close()
    preprocessor = Preprocessor(data)
    preprocessed_documents = preprocessor.preprocess()
    with open('data/preprocessed.json', 'w') as f:
        f.write(json.dumps(preprocessed_documents, indent=4))
        f.close()
