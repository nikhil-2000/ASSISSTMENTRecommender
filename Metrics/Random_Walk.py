import numpy as np
import pandas as pd

from Objects.compute_embeddings import CalcEmbeddings

"""
For each model:
 - A ranking given a set of problems
 - Nearby Questions given an anchor
 
"""


class RandomWalkMetrics:

    def __init__(self, filename, model_file, size=0):
        self.embedder = CalcEmbeddings(filename, model_file, size=size)

        self.embeddings, self.metadata, self.e_dict = self.embedder.get_embeddings()
        self.metadata = pd.DataFrame(self.metadata, columns=["problem_id", "skill_id", "skill_name"])

        self.hits = 0
        self.ranks = []

    def top_n_questions(self, anchor, search_size):
        # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])
        anchor_id = anchor.problem_id.item()
        anchor_embedding = self.e_dict[anchor_id]
        dists = np.linalg.norm(self.embeddings - anchor_embedding, axis=1)
        sorted_indexes = np.argsort(dists)
        best_indexes = sorted_indexes[1:search_size + 1]

        if len(best_indexes) >= search_size:
            return self.metadata.iloc[best_indexes]["problem_id"].to_list()
        else:
            return self.metadata.iloc[best_indexes]["problem_id"].to_list() + [0] * (search_size - len(best_indexes))

    def rank_questions(self, ids, anchor):
        anchor_id, anchor_skill = anchor.problem_id.item(), anchor.skill_id.item()
        anchor = self.e_dict[anchor_id]

        embeddings_to_rank = [self.e_dict[k] for k in ids]
        embeddings_to_rank = np.array(embeddings_to_rank).squeeze()

        dists = np.linalg.norm(embeddings_to_rank - anchor, axis=1)
        sorted_indexes = np.argsort(dists)

        return np.array(ids)[sorted_indexes].tolist()

    def hitrate(self, tests):
        return 100 * self.hits/ tests

    def mean_rank(self):
        return sum(self.ranks) / len(self.ranks)