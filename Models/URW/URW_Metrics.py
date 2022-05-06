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
from Models.MetricBase import MetricBase
from Models.URW.URW import UnweightedRandomWalk
from datareader import Datareader
from helper_funcs import MRR, convert_distances

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


class URW_Metrics(MetricBase):

    def __init__(self, dataset, model):
        self.dataset = dataset

        self.model = model
        super(URW_Metrics, self).__init__()


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
            if max_qs > 0:
                predicted = user.get_questions_by_correct(anchor, max_qs)
                # predicted = user.get_questions(anchor, max_qs)
                # predicted = user.get_questions_by_skill(anchor.skill_id, max_qs)
                top_n.extend(predicted)
                questions_to_choose[i] -= len(predicted)

            if i+1 < len(users):
                leftover = questions_to_choose[i]
                questions_to_choose[i+1] += leftover
                questions_to_choose[i] = 0

        remaining = sum(questions_to_choose)
        break_loop = 0
        while remaining > 0:
            q_id = self.dataset.item_ids.sample(1).item()
            if not q_id in top_n:
                top_n.append(q_id)
                remaining -= 1
            else:
                break_loop += 1

            if break_loop > 50:
                top_n.extend([0] * remaining)
                remaining = -1

        return top_n

    # def probability_selection(self, weights, closest_users):


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







