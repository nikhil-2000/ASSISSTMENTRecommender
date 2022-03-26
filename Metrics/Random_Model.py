import pandas as pd
import random


class RandomChoiceMetrics:

    def __init__(self, metadata):

        self.metadata = metadata
        # self.metadata = pd.DataFrame(self.metadata, columns=["problem_id", "skill_id", "skill_name"])
        self.hits = 0
        self.ranks = []

    def top_n_questions(self, anchor, search_size):
        # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])
        anchor_id = anchor.problem_id.item()
        if search_size <= len(self.metadata):
            return self.metadata.problem_id.sample(search_size).to_list()
        else:
            metadata_size = len(self.metadata)
            return self.metadata.sample(metadata_size).to_list() + [0] * (search_size - metadata_size)

    def rank_questions(self, ids, anchor):
        random.shuffle(ids)
        return ids

    def hitrate(self, tests):
        return 100 * self.hits / tests

    def mean_rank(self):
        return sum(self.ranks) / len(self.ranks)
