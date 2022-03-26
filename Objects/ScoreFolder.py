import random

import torch

from Objects.Data_reader import Dataset
import pandas as pd


class ScoreFolder:
    def __init__(self, filename, size = 0):
        self.dataset = Dataset(filename, size=size)
        self.interactions_df = self.dataset.interactions
        self.users_df = self.dataset.users
        self.problems_df = self.dataset.problems

        self.memory = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        problem_id = self.problems_df.index[index]
        skill_id, skill_name = self.map_id_to_name(problem_id)

        features = self.get_features(problem_id)
        return features, skill_id, skill_name, problem_id

    def map_id_to_name(self, q_id):
        condition = self.interactions_df["problem_id"] == q_id
        df = self.interactions_df.loc[condition][["skill_name", "skill_id"]]
        potential_names_ids = list(df.itertuples(index=False, name=None))
        if not potential_names_ids:
            name = "Unknown"
            skill_id = 0
        else:
            name, skill_id = random.choice(potential_names_ids)

        return name, skill_id

    def get_features(self, problem_id):

        vector = self.problems_df.loc[problem_id].to_list()

        # condition = self.problems["problem_id"] == q_id
        # row = self.problems.loc[condition]
        # return torch.Tensor(row.values[0][1:])
        return torch.Tensor(vector)

