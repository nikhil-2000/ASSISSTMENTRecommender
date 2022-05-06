import torch

from Datasets.datasetBase import DatasetBase
from datareader import Datareader
from helper_funcs import vector_features, add_metrics


class TrainDataset(DatasetBase):
    def __init__(self, interactions):
        # self.dataset = Dataset(filename, size=size)

        super(TrainDataset, self).__init__(interactions)


    def __getitem__(self, index: int):
        # Extracts Categories + Metrics
        s = self.items_df.iloc[index]
        question_data = s[vector_features].to_list()
        user_data = s[vector_features].to_list()
        data = user_data + question_data

        skill_id = s["skill_id"].item()
        user_id = self.items_df.index.get_level_values(1)[index]

        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])

        return torch.Tensor(data), int(skill_id), int(user_id)


if __name__ == '__main__':
    reader = Datareader("../skill_builder_data.csv", size = 1000)
    mt = TrainDataset(reader.interactions)
    df = mt.items_df.reset_index()
    row = df.iloc[0]

    print()
