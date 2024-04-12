import os
from typing import List

import numpy as np
import wandb


class Evaluation:

    def __init__(self, name: str):
        self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0
        for i in range(len(predicted)):
            tp = len(set(predicted[i]).intersection(set(actual[i])))
            precision += tp / len(predicted[i])
        precision /= len(predicted)
        return precision

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0
        for i in range(len(predicted)):
            tp = len(set(predicted[i]).intersection(set(actual[i])))
            recall += tp / len(actual[i])
        recall /= len(predicted)
        return recall

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        return (2 * precision * recall) / (precision + recall)

    def calculate_AP(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Average Precision of the predicted results

        Parameters
        ----------
        actual : List[str]
            The actual results
        predicted : List[str]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0
        correct_count = 0
        for i in range(len(predicted)):
            if predicted[i] in actual:
                correct_count += 1
                AP += correct_count / (i + 1)
        AP /= correct_count
        return AP

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0
        for i in range(len(predicted)):
            MAP += self.calculate_AP(actual[i], predicted[i])
        MAP /= len(predicted)
        return MAP

    def calculate_DCG(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Discounted Cumulative Gain (DCG) of the predicted results

        Parameters
        ----------
        actual : List[str]
            The actual results
        predicted : List[str]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0
        relevance = {}
        for i in range(len(actual)):
            relevance[actual[i]] = len(actual) - i
        for i in range(len(predicted)):
            if predicted[i] in relevance:
                DCG += relevance.get(predicted[i]) / np.log2(i + 1)
        return DCG

    def calculate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0
        for i in range(len(predicted)):
            max_score = self.calculate_DCG(actual[i], actual[i])
            dcg = self.calculate_DCG(actual[i], predicted[i])
            NDCG += dcg / max_score
        NDCG /= len(predicted)
        return NDCG

    def calculate_RR(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[str]
            The actual results
        predicted : List[str]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        for i in range(len(predicted)):
            if predicted[i] in actual:
                return 1 / (i + 1)
        return 0.0

    def calculate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0
        for i in range(len(predicted)):
            MRR += self.calculate_RR(actual[i], predicted[i])
        MRR /= len(predicted)
        return MRR

    def print_evaluation(self, precision, recall, f1, map, ndcg, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        print(f'Precision = {precision}')
        print(f'Recall    = {recall}')
        print(f'F1        = {f1}')
        print(f'MAP       = {map}')
        print(f'NDCG      = {ndcg}')
        print(f'MRR       = {mrr}')

    def log_evaluation(self, precision, recall, f1, map, ndcg, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=f'evaluation result {self.name}')
        wandb.log(
            {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'map': map,
                'ndcg': ndcg,
                'mrr': mrr
            }
        )

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        ndcg = self.calculate_NDCG(actual, predicted)
        mrr = self.calculate_MRR(actual, predicted)

        # call print and viualize functions
        self.print_evaluation(precision, recall, f1, map_score, ndcg, mrr)
        self.log_evaluation(precision, recall, f1, map_score, ndcg, mrr)
