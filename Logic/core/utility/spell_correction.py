class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        shingles_count = len(word) - k + 1
        for i in range(shingles_count):
            shingles.add(word[i:i + k])
        return shingles

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """
        return len(first_set.intersection(second_set)) / len(first_set.union(second_set))

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()
        for doc_id in all_documents:
            doc_details = ''
            document = all_documents[doc_id]
            for summary in document['summaries']:
                s = summary.strip()
                doc_details = doc_details + s + ' '
            for star in document['stars']:
                s = star.strip()
                doc_details = doc_details + s + ' '
            for genre in document['genres']:
                s = genre.strip()
                doc_details = doc_details + s + ' '
            words = doc_details.split()
            for word in words:
                if word not in all_shingled_words:
                    all_shingled_words[word] = self.shingle_word(word)
                    word_counter[word] = 1
                else:
                    word_counter[word] += 1
        return all_shingled_words, word_counter

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = {}
        candidates = {}
        shingles = self.shingle_word(word)
        for candidate, candidate_shingle in self.all_shingled_words.items():
            candidates[candidate] = self.jaccard_score(shingles, candidate_shingle)

        if len(candidates.keys()) > 5:
            for i in range(5):
                max_key = max(candidates, key=candidates.get)
                top5_candidates[max_key] = candidates[max_key]
                candidates.pop(max_key, None)
        else:
            for key in candidates.keys():
                top5_candidates[key] = candidates[key]

        max_tf = -1
        for word in top5_candidates:
            max_tf = max(max_tf, self.word_counter[word])

        for key in top5_candidates:
            top5_candidates[key] = top5_candidates[key] * self.word_counter[key] / max_tf
        top5_candidates = dict(sorted(top5_candidates.items(), key=lambda x: x[1], reverse=True))

        return [word for word in top5_candidates.keys()]

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ''
        query_words = query.split()
        for word in query_words:
            if word in self.all_shingled_words:
                final_result = final_result + word + ' '
            else:
                candidates = self.find_nearest_words(word)
                if candidates[0] is not None:
                    final_result = final_result + candidates[0] + ' '
                else:
                    final_result = final_result + word + ' '
        return final_result
