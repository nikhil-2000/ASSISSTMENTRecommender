import random

import torch
import numpy as np
import pandas as pd

from Datasets.datasetBase import DatasetBase
from datareader import Datareader
from helper_funcs import vector_features, add_metrics


class TestDataset(DatasetBase):
    def __init__(self, interactions):
        # self.dataset = Dataset(filename, size=size)

        super(TestDataset, self).__init__(interactions)
        _ids = self.interaction_df.skill_id.tolist()
        _names = self.interaction_df.skill_name.tolist()
        key_vals = zip(_ids, _names)
        self.id_to_name = dict(key_vals)


    def __getitem__(self, index: int):
        # Extracts Categories + Metrics
        interaction = self.items_df.iloc[index]
        item_id = self.items_df.index.get_level_values(0)[index]
        user_id = self.items_df.index.get_level_values(1)[index]


        question_data = interaction[vector_features].to_list()
        user_data = self.users_df.loc[user_id][vector_features].to_list()
        data = user_data + question_data
        skill_id = interaction["skill_id"]
        skill_name = self.id_to_name[skill_id]
        skill_info = [skill_id,skill_name]
        metadata = [user_id] + [item_id] + skill_info + data


        return torch.Tensor(data), int(skill_id), metadata, (item_id, user_id)

    def get_by_ids(self, user_id, item_id):

        interaction = self.items_df.loc[item_id, user_id]

        question_data = interaction[vector_features].to_list()
        user_data = self.users_df.loc[user_id][vector_features].to_list()
        data = user_data + question_data
        skill_id = interaction["skill_id"]

        return torch.Tensor(data), torch.Tensor([skill_id]).long().squeeze()

if __name__ == '__main__':
    datareader = Datareader("../skill_builder_data.csv", size =0)
    mt = TestDataset(datareader.interactions)
    print(mt[0])

