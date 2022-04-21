import random

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from Datasets.Training import TrainDataset
from datareader import Datareader


class UnweightedRandomWalk:

    def __init__(self, dataset: TrainDataset, closest =50):

        self.dataset = dataset
        self.users, self.skills = self.create_users()
        self.vectors, self.id_to_idx = self.get_user_scores()
        self.dists = pairwise_distances(self.vectors.transpose())
        self.closest = closest

    def create_users(self):
        interactions = self.dataset.interaction_df
        user_performance = interactions.groupby(["user_id", "skill_id"])["correct"].mean()
        user_performance = pd.DataFrame(user_performance)
        user_performance.reset_index(inplace=True)

        all_users = user_performance.user_id.unique()
        all_skills = user_performance.skill_id.unique()
        all_questions = self.dataset.item_ids.unique()

        users = []
        # print(len(all_skills))
        # skill_vectors = pd.DataFrame(columns=["user_id"] + list(all_skills))
        print("Creating Users")
        for user in tqdm(all_users):
            user_scores = user_performance[user_performance["user_id"] == user]
            user_interactions = interactions[interactions.user_id == user]
            new_user = User(user_scores, all_skills, user_interactions)
            # new_user.generate_q_completed_df(all_questions)
            users.append(new_user)
        # user_performance.reset_index(inplace = True)

        # users = list(filter(lambda x: x.has_skills(), users))

        return users, all_skills

    def get_user_scores(self):
        score_matrix = np.zeros((len(self.skills), len(self.users)))
        ids_to_idx = {}
        for idx, user in enumerate(self.users):
            ids_to_idx[user.id] = idx
            score_matrix[:, idx] = user.get_vector()

        return score_matrix, ids_to_idx

    def get_closest_users(self, user_id):
        if user_id not in self.id_to_idx:
            user_id = random.choice(self.users).id
            return self.get_closest_users(user_id)

        idx = self.id_to_idx[user_id]
        dists_from_users = self.dists[:, idx]
        dists_from_users = dists_from_users[dists_from_users > 0]
        sorted_indexes = np.argsort(dists_from_users)
        closest = max(len(sorted_indexes) // self.closest, 10)
        best_idxs = sorted_indexes[:closest]
        closest_users_dists = dists_from_users[best_idxs]
        closest_users = [self.users[i] for i in best_idxs]
        return closest_users_dists, closest_users


class User:

    def __init__(self, subset_df: pd.DataFrame, all_skills, user_interactions: pd.DataFrame):
        self.id = subset_df.iloc[0]["user_id"]
        d = {k: 0 for k in all_skills}
        subset_df = subset_df.sort_values(by=["skill_id"])
        for i, row in subset_df.iterrows():
            skill, score = row["skill_id"], row["correct"]
            d[skill] = score

        # if sum(d.values()) == 0:
        #     self.scores = np.array([0.5 for c in all_skills])


        self.scores = np.array(list(d.values()))
        self.interactions = user_interactions

        self.user_qs = self.interactions.problem_id.unique()
        self.is_completed = pd.DataFrame(columns=["problem_id", "attempted"])

    def generate_q_completed_df(self, all_question_ids):
        self.is_completed.problem_id = all_question_ids
        self.is_completed.attempted = self.is_completed.problem_id.isin(self.user_qs).astype(int)
        self.is_completed.set_index("problem_id", inplace=True)

    def get_vector(self):
        # print(self.scores.transpose().shape)
        return self.scores.transpose()

    def has_skills(self):
        return any(self.scores)

    def get_questions_by_skill(self, skill_id, n):
        qs = self.interactions[self.interactions.skill_id == skill_id]
        remaining = n - len(qs)
        if len(self.interactions) >= remaining > 0:
            random_qs = self.interactions.sample(remaining)
            return qs.problem_id.to_list() + random_qs.problem_id.to_list()
        elif len(self.interactions) < remaining:
            random_qs = self.interactions
            zeroes = [0] * (n - len(random_qs) - len(qs))
            return qs.problem_id.to_list() + random_qs.problem_id.to_list() + zeroes
        else:
            return qs.sample(n).problem_id.to_list()

    def get_questions_by_correct(self, anchor, max_ret):
        correct = anchor.correct.item()
        potential_qs = self.interactions.loc[self.interactions.correct == correct]

        if len(potential_qs) > max_ret:
            return potential_qs.problem_id.sample(max_ret).to_list()
        else:
            return potential_qs.problem_id.to_list()

    def get_correct_ids(self, questions):
        attempted = [1 if q in self.user_qs else 0 for q in questions]
        return np.array(attempted).squeeze()


if __name__ == '__main__':
    datareader = Datareader("../../skill_builder_data.csv", size=0)
    train = TrainDataset(datareader.train)

    urw = UnweightedRandomWalk(train)

    id = train.interaction_df.sample(1).user_id.item()
    dists, users = urw.get_closest_users(id)
