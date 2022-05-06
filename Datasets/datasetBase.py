import random

import torch

from helper_funcs import vector_features, add_metrics
import pandas as pd


class DatasetBase:
    def __init__(self, interactions):
        # self.dataset = Dataset(filename, size=size)
        self.interaction_df = interactions
        self.items_df, self.users_df = add_metrics(interactions)
        self.users = self.create_users()
        self.item_ids = self.items_df.index.get_level_values(0).unique().to_series()


    def __len__(self):
        return len(self.items_df)


    def sample_user(self):
        return random.choice(self.users)

    def create_users(self):
        interactions = self.interaction_df

        user_ratings = interactions.sort_values("user_id")
        all_users = user_ratings.user_id.unique()
        users = []
        # print(len(all_skills))
        # skill_vectors = pd.DataFrame(columns=["user_id"] + list(all_skills))
        # print("\nCreating Users")
        # for user in tqdm(all_users):
        for user in all_users:
            u_ratings = user_ratings[user_ratings["user_id"] == user]
            new_user = User(u_ratings)
            users.append(new_user)

        return users


class User:

    def __init__(self, subset_df: pd.DataFrame):
        self.id = subset_df.iloc[0]["user_id"]
        self.interactions = subset_df

        self.user_qs = self.interactions.problem_id.unique()
        self.is_completed = pd.DataFrame(columns=["problem_id", "attempted"])

    def generate_q_completed_df(self, all_question_ids):
        self.is_completed.problem_id = all_question_ids
        self.is_completed.attempted = self.is_completed.problem_id.isin(self.user_qs).astype(int)
        self.is_completed.set_index("problem_id", inplace=True)
