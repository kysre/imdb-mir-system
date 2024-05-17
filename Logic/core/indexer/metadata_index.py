from .index_reader import Index_reader
from .indexes_enum import Indexes, Index_types
import json


class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        self.documents = self.read_documents()
        self.metadata_index = {}

    def read_documents(self):
        """
        Reads the documents.
        
        """
        with open('data/preprocessed.json', 'r') as f:
            data = json.loads(f.read())
            f.close()
        return data

    def create_metadata_index(self):
        """
        Creates the metadata index.
        """
        self.metadata_index = {
            'average_document_length': {
                'stars': self.get_average_document_field_length('stars'),
                'genres': self.get_average_document_field_length('genres'),
                'summaries': self.get_average_document_field_length('summaries')
            },
            'document_count': len(self.documents)
        }

    def get_average_document_field_length(self, where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        length = 0
        for doc in self.documents:
            if where != 'summaries':
                length += len(doc[where])
            else:
                for summary in doc['summaries']:
                    length += len(summary.split())
        return length / len(self.documents)

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path = path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


if __name__ == "__main__":
    metadata_index = Metadata_index()
    metadata_index.create_metadata_index()
    metadata_index.store_metadata_index('data/index/')
