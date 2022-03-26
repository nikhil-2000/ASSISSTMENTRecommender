import pandas as pd
import random

class SameSkillMetrics:

    def __init__(self, metadata):

        self.metadata = metadata
        # self.metadata = pd.DataFrame(self.metadata, columns=["problem_id", "skill_id", "skill_name"])

        self.hits = 0
        self.ranks = []


    def top_n_questions(self, anchor, search_size):
        # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])
        anchor_id = anchor.problem_id.item()
        condition = self.metadata["problem_id"] == anchor_id
        skill = self.metadata[condition].skill_id.to_list()[0]

        same_skill_mask = self.metadata.skill_id == skill
        same_skill_qs = self.metadata.loc[same_skill_mask].problem_id
        other_qs = self.metadata.loc[~same_skill_mask].problem_id
        remaining = search_size - len(same_skill_qs)

        if len(other_qs) >= remaining > 0:
            random_qs = other_qs.sample(remaining)
            return same_skill_qs.to_list() + random_qs.to_list()
        elif len(self.metadata) < remaining:
            zeroes = [0] * (search_size - len(other_qs) - len(same_skill_qs))
            return same_skill_qs.to_list() + other_qs.to_list() + zeroes
        else:
            return same_skill_qs.sample(search_size).to_list()

    def rank_questions(self, ids, anchor):
        anchor_id, anchor_skill = anchor.problem_id.item(), anchor.skill_id.item()
        ids = list(ids)
        same_skill_qs = set(self.metadata.loc[self.metadata["skill_id"] == anchor_skill].problem_id)
        in_samples = set(ids).intersection(same_skill_qs)
        ordering = []
        for id in in_samples:
            ordering.append(id)
            ids.remove(id)

        return ordering + ids

    def hitrate(self, tests):
        return 100 * self.hits/ tests

    def mean_rank(self):
        return sum(self.ranks) / len(self.ranks)
