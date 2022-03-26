from __future__ import absolute_import

import os
import random
import sys

import networkx as nx

from tqdm import tqdm

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)

import torch
import pandas as pd
from Objects.Data_reader import Dataset
import numpy as np

class TriplesGenerator:

    def __init__(self, filename, is_training=True, size=0):

        self.dataset = Dataset(filename, size=size)
        self.interactions_df = self.dataset.interactions
        self.users_df = self.dataset.users
        self.problems_df = self.dataset.problems

        self.memory = {}

        self.is_training = is_training
        self.gen_graph()

    def __len__(self):
        return len(self.dataset)

    def gen_graph(self):

        B = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        user_data = self.users_df.to_dict('records')
        user_nodes = []
        for i,data in enumerate(user_data):
            id = self.dataset.user_ids.iloc[i]
            user_nodes.append((id, data))

        B.add_nodes_from(user_nodes, bipartite=0)

        problem_nodes = []
        problem_data = self.problems_df.to_dict('records')
        for i,data in enumerate(problem_data):
            id = self.dataset.problem_ids.iloc[i]
            problem_nodes.append((id, data))
        B.add_nodes_from(problem_nodes, bipartite=1)

        if self.is_training:
            for i, row in self.interactions_df.iterrows():
                user, question = row["user_id"], row["problem_id"]
                B.add_edge(user, question)

        print("Nodes: ", len(B.nodes))
        print("Edges: ", len(B.edges))

        self.graph = B

    def __getitem__(self, index):
        anchor_id = self.problems_df.index[index]

        # random walk from current question
        current_question = anchor_id
        dict_counter = dict()
        dict_counter[current_question] = 1

        # Traversing through the neighbors of start node
        for i in range(100):
            list_for_users = list(self.graph.neighbors(current_question))
            if len(list_for_users) == 0:  # if random_node having no outgoing edges
                print("No user attempts")
            else:
                random_user = random.choice(list_for_users)  # choose a user randomly from neighbors
                list_for_questions = list(self.graph.neighbors(random_user))
                next_question = random.choice(list_for_questions)
                if next_question in dict_counter:
                    dict_counter[next_question] = dict_counter[next_question] + 1
                else:
                    dict_counter[next_question] = 1

                current_question = next_question

        del dict_counter[current_question]
        counts = list(dict_counter.items())
        by_visits = sorted(counts, key=lambda x: x[1], reverse=True)
        by_visits = filter(lambda q: q[0] in self.dataset.problem_ids, by_visits)
        by_visits = list(by_visits)
        positives = by_visits[:10]
        negatives = by_visits[10:]

        assert all([q in self.dataset.problem_ids for q,_ in by_visits])

        if len(positives) == 0:
            positive = self.dataset.problem_ids.sample(1).squeeze()
        else:
            positive, _ = random.choice(positives)

        if len(negatives) == 0:
            negative = self.dataset.problem_ids.sample(1).squeeze()
        else:
            negative, _ = random.choice(negatives)

        features = list(map(self.get_features, [anchor_id, positive, negative]))

        if current_question == positive:
            print("Same Pos")
        if current_question == negative:
            print("Same Neg")
        if negative == positive:
            print("Pos == Neg")
        return features

        # return current_movie, positive.squeeze(), negative.squeeze()

    def get_features(self, problem_id):

        vector = self.problems_df.loc[problem_id].to_list()

        # condition = self.problems["problem_id"] == q_id
        # row = self.problems.loc[condition]
        # return torch.Tensor(row.values[0][1:])
        return torch.Tensor(vector)


if __name__ == '__main__':
    tl = TriplesGenerator("../skill_builder_data.csv", size = 1000)
    a, p , n = tl.__getitem__(1)

    print(a)
    print(p)
    print(n)
