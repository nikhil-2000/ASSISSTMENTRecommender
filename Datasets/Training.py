import torch

from helper_funcs import vector_features, add_metrics


class TrainDataset:
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
        data = s[vector_features].to_list()

        skill_id = s["skill_id"].item()

        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])

        return torch.Tensor(data), skill_id


if __name__ == '__main__':

    mt = TrainDataset("../../../ml-100k/ua.base")
    print(mt[0])
