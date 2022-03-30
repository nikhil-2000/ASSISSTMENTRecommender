import random
import time

import pandas as pd
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from Models.URW.URW import UnweightedRandomWalk, User
from datareader import Datareader

"""
Create a user similarity matrix
Get the anchor and performance
Top N 
Get the K similar users, and find problems which they have a similar performance on
 - Prioritise similar skill items
Make a weighted choice of users, pick more problems from more similar users

Ranking Questions
Given 1000 ids
Get the K nearest users, rank by most completed problems
"""


class URW_Metrics:

    def __init__(self, dataset, model):
        self.dataset = dataset

        self.model = model

        self.hits = 0
        self.ranks = []

    def __len__(self):
        return len(self.dataset.ratings_df)

    def sample_user(self):
        return random.choice(self.model.users)

    def hitrate(self, tests):
        return 100 * self.hits / tests

    def mean_rank(self):
        return sum(self.ranks) / len(self.ranks)

    def top_n_questions(self, anchor, search_size):
        # Get top 10% closest users
        # Assign each user a weight
        # Pick problems based on weight
        # Each question chosen should have the user performance be similair
        # i.e if anchor is correct, pick other correct problems from similar users

        top_n = []

        user_id = anchor.user_id.item()
        distances, users = self.model.get_closest_users(user_id)

        questions_to_choose = convert_distances(distances, search_size)
        questions_to_choose = questions_to_choose.astype(int)

        for i, user in enumerate(users):
            max_qs = questions_to_choose[i]
            predicted = user.get_questions_by_correct(anchor, max_qs)
            top_n.extend(predicted)
            questions_to_choose[i] -= len(predicted)

        remaining = sum(questions_to_choose)
        break_loop = 0
        while remaining > 0:
            id = self.dataset.item_ids.sample(1).item()
            if not id in top_n:
                top_n.append(id)
                remaining -= 1
            else:
                break_loop += 1

            if break_loop > 50:
                top_n.extend([0] * remaining)
                remaining = -1

        return top_n

    def rank_questions(self, ids, anchor):
        user_id = anchor.user_id.item()
        distances, users = self.model.get_closest_users(user_id)
        # t_1 = time.time()

        weights = convert_distances(distances, 100) / 100

        # t_2 = time.time()

        # for problem in ids:
        #     user_counts = [u.attempted_q(problem) * weights[i] for i, u in enumerate(closest_users)]
        #     problem_id_to_count[problem] = sum(user_counts)
        all_correct = np.zeros((len(users), len(ids)))
        for i, user in enumerate(users):
            user_performance = user.get_correct_ids(ids) * weights[i]
            all_correct[i] = user_performance
        # t_3 = time.time()

        all_correct = all_correct.sum(axis=0).tolist()
        # t_4 = time.time()

        counts = list(zip(ids, all_correct))
        highest = sorted(counts, key=lambda x: x[1], reverse=True)
        # t_5 = time.time()

        # ts = np.array([t_1 - start, t_2-t_1, t_3-t_2, t_4-t_3,t_5-t_4])
        # self.times[self.i] = ts
        # self.i += 1

        return [h[0] for h in highest]

def convert_distances(distances, search_size):
    x = 1 / distances
    x = x / x.sum(axis=0)
    x = search_size * x
    x = np.floor(x)
    leftover = search_size - sum(x)
    x[0] += leftover
    return x


def test_model(file):
    train_reader = Datareader(file, size=10000, training_frac=0.7)

    metrics = []
    datasets = []

    data = TrainDataset(train_reader.train)
    urw = UnweightedRandomWalk(data)

    for d in [train_reader.train, train_reader.test, train_reader.interactions]:

        metric = URW_Metrics(data, urw)
        metrics.append(metric)
        datasets.append(data)

    model_names = ["Train", "Test", "All"]

    params = zip(metrics, datasets, model_names)

    search_size = 30
    tests = 1000
    samples = 1000
    output = PrettyTable()
    output.field_names = ["Data", "Hitrate", "Mean Rank"]

    ranks = []
    for metric, data, name in params:

        print("\nTesting " + name)
        for i in trange(tests):
        # for i in range(tests):
            # Pick Random User
            total_interactions = 0
            while total_interactions < 5:
                user = metric.sample_user()
                user_interactions, total_interactions = user.interactions, len(user.interactions)
            # Generate Anchor Positive
            a_idx, p_idx = random.sample(range(0, total_interactions), 2)
            anchor = user_interactions.iloc[a_idx]

            positive = user_interactions.iloc[p_idx]
            positive_id = positive.problem_id.item()

            without_positive = data.item_ids[~data.item_ids.isin(user_interactions.problem_id.unique())]
            random_ids = np.random.choice(without_positive, samples).tolist()
            all_ids = random_ids + [positive_id]
            random.shuffle(all_ids)

            # Find n Closest
            top_n = metric.top_n_questions(anchor, search_size)
            ranking = metric.rank_questions(all_ids, anchor)

            set_prediction = set(top_n)
            if positive_id in set_prediction:
                metric.hits += 1

            rank = ranking.index(positive_id)

            metric.ranks.append(rank)

        hr = metric.hitrate(tests)
        mr = metric.mean_rank()
        output.add_row([name, hr, mr])
        ranks.append(mr)
        print(output)

    return output, ranks


if __name__ == '__main__':
    out, ranks = test_model("../../skill_builder_data.csv")
    print(out)
    print(ranks)
