from preprocess import Preprocessor


class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """
        preprocessor = Preprocessor(documents=[])
        query = preprocessor.normalize(query)
        return query

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Shawshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        query = self.remove_stop_words_from_query(query)

        doc_words = doc.split()
        query_words = query.split()
        not_exist_words_set = set(query_words)
        chosen_words = [0] * len(doc_words)

        for i, word in enumerate(doc_words):
            if word in query_words:
                chosen_words[i] = 1
                not_exist_words_set.discard(word)

        for i, val in enumerate(chosen_words):
            if val == 1:
                j = 1
                while j <= self.number_of_words_on_each_side:
                    if i + j < len(doc_words) and chosen_words[i + j] != 1:
                        chosen_words[i + j] = 2
                    if i - j >= 0 and chosen_words[i - j] != 1:
                        chosen_words[i + j] = 2
                    j += 1

        snippet_words = []
        for i, word in enumerate(doc_words):
            if chosen_words[i] != 0:
                snippet_words.append(word)

        not_exist_words = list(not_exist_words_set)
        final_snippet = ' '.join(snippet_words)

        return final_snippet, not_exist_words
