import math

import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            posting_list = self.index.get(term, {})
            df = len(posting_list.keys())
            if df == 0:
                idf = 0
            else:
                idf = np.log10(self.N / df)
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        tfs = {}
        for term in query:
            if term not in tfs.keys():
                tfs[term] = 0
            tfs[term] += 1
        return tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        documents = self.get_list_of_documents(query)
        document_method, query_method = method.split('.')
        query_tfs = self.get_query_tfs(query)
        document_scores = {}
        for document_id in documents:
            if document_id not in document_scores:
                document_scores[document_id] = self.get_vector_space_model_score(
                    query, query_tfs, document_id, document_method, query_method
                )
        return document_scores

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        query_vector = self.get_vector_space(query_tfs.copy(), query_method)

        document_vector = {}
        for term in query_vector:
            posting_list = self.index.get(term, {})
            if document_id in posting_list:
                document_vector[term] = posting_list[document_id]
            else:
                document_vector[term] = 0
        document_vector = self.get_vector_space(document_vector, document_method)

        score = 0
        for term in query_vector:
            score += query_vector[term] * document_vector[term]
        return score

    def get_vector_space(self, vector, method):
        if method[0] == 'l':
            for term in vector:
                vector[term] = 1 + np.log10(vector[term]) if vector[term] != 0 else 0
        if method[1] == 't':
            for term in vector:
                vector[term] *= self.get_idf(term)
        if method[2] == 'c':
            magnitude = np.sqrt(np.sum(np.array(list(vector.values())) ** 2))
            for term in vector:
                vector[term] /= magnitude
        return vector

    def compute_scores_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        documents = self.get_list_of_documents(query)
        document_scores = {}
        for document_id in documents:
            document_scores[document_id] = self.get_okapi_bm25_score(
                query, document_id, average_document_field_length, document_lengths
            )
        return document_scores

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        k = 1.2
        b = 0.75
        scores = {}

        for term in query:
            scores[term] = self.get_idf(term)

        for term in scores:
            posting_list = self.index.get(term, {})
            scores[term] *= self.get_bm25_multiple(
                k=k,
                b=b,
                tf=posting_list.get(document_id, 0),
                document_length=document_lengths[document_id],
                average_document_field_length=average_document_field_length
            )

        return np.sum(np.array(list(scores.values())))

    def get_bm25_multiple(self, k, b, tf, document_length, average_document_field_length):
        return ((k + 1) * tf) / (k * ((1 - b) + b * (document_length / average_document_field_length)) + tf)
