import random

import torch
import numpy as np
import pandas as pd
from helper_funcs import vector_features, add_metrics


class TestDataset:
    def __init__(self, interactions):
        # self.dataset = Dataset(filename, size=size)
        self.interaction_df = interactions
        self.item_df, self.users_df = add_metrics(interactions)


        self.item_ids = self.item_df.index.to_series()


    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, index: int):
        # Extracts Categories + Metrics
        item_id = self.item_ids.iloc[index]
        s = self.item_df.loc[item_id]
        skill_name, skill_id = self.map_id_to_name(item_id)
        data = s[vector_features].to_list()

        return torch.Tensor(data), (skill_id, skill_name, item_id)


    def map_id_to_name(self, q_id):
        condition = self.interaction_df["problem_id"] == q_id
        df = self.interaction_df.loc[condition][["skill_name", "skill_id"]]
        potential_names_ids = list(df.itertuples(index=False, name=None))
        if not potential_names_ids:
            name = "Unknown"
            skill_id = 0
        else:
            name, skill_id = random.choice(potential_names_ids)


        return name, skill_id
