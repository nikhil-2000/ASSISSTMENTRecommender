import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import random
from itertools import permutations
import math

import torch
from torch.utils.data import DataLoader


class AssignmentDataset:

    def __init__(self, isTraining = True, training_data = True):
        interactions, users, questions = get_dfs(1000, False, training_data)

        graph = gen_graph(interactions, users, questions, add_edges = isTraining)
        self.graph = graph

        self.interactions, self.users, self.questions =  interactions, users, questions


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        current_question = self.questions.loc[index]["problem_id"]

        # random walk from current question
        dict_counter = dict()
        dict_counter[current_question] = 1

        # Traversing through the neighbors of start node
        for i in range(1000):
            neighbours = list(self.graph.neighbors(current_question))
            if len(neighbours) == 0:  # if random_node having no outgoing edges
                print("No other qs in assignment")
            else:
                next_question = random.choice(neighbours)
                if next_question in dict_counter:
                    dict_counter[next_question] = dict_counter[next_question] + 1
                else:
                    dict_counter[next_question] = 1

        del dict_counter[current_question]
        counts = list(dict_counter.items())
        by_visits = sorted(counts, key=lambda x: x[1], reverse=True)
        positives = by_visits[:10]
        negatives = by_visits[10:20]

        if len(positives) == 0:
            positive = random.choice(self.questions["problem_id"].to_list())
        else:
            positive, _ = random.choice(positives)

        if len(negatives) == 0:
            negative = random.choice(self.questions["problem_id"].to_list())
        else:
            negative, _ = random.choice(negatives)

        features = list(map(self.get_features, [current_question, positive, negative]))

        if current_question == positive:
            print("Same Pos")
        if current_question == negative:
            print("Same Neg")
        if negative == positive:
            print("Pos == Neg")
        return features

    def get_features(self, q_id):
        features = ["attempt_count", "correct", "ms_first_response"]
        vector = []
        node_data = self.graph.nodes[q_id]
        for f in features:
            vector.append(node_data[f])
        # condition = self.problems["problem_id"] == q_id
        # row = self.problems.loc[condition]
        # return torch.Tensor(row.values[0][1:])
        return torch.Tensor(vector)

class EmbeddingDataset(AssignmentDataset):

    def __init__(self, isTraining=False, training_data = True):
        super().__init__(isTraining=isTraining, training_data= training_data)
        self.interactions["skill_id"] = self.interactions["skill_id"].apply(lambda x : int(x))


    def __getitem__(self, index):
        q_id = self.questions.loc[index]["problem_id"]
        skill_id, skill_name = self.map_id_to_name(q_id)

        features = self.get_features(q_id)
        return features,skill_id, skill_name, q_id

    def map_id_to_name(self,q_id):
        condition = self.interactions["problem_id"] == q_id
        potential_names = self.interactions.loc[condition]["skill_name"].to_list()
        potential_ids = self.interactions.loc[condition]["skill_id"].to_list()
        if not potential_names:
            name = "Unknown"
        else:
            name = random.choice(potential_names)
        if not potential_ids:
            skill_id = 0
        else:
            skill_id = int(random.choice(potential_ids))

        return name, skill_id



def normalise_df(df):
    return (df - df.min()) / (df.max() - df.min())


def get_dfs(n = 1, subset_data = False, training_data = True):
    if training_data:
        data = pd.read_csv("../skill_builder_data.csv", encoding='latin-1')
    else:
        data = pd.read_csv("../non_skill_builder_data_new.csv", encoding='latin-1')

    if subset_data:
        data = data.head(n)
    data['skill_id'] = data['skill_id'].fillna(0)
    data["skill_id"] = data["skill_id"].apply(getSkillID)
    data["skill_id"] = data["skill_id"].apply(int)

    data['skill_name'] = data['skill_name'].fillna("Unknown")
    data["skill_name"] = data["skill_name"].apply(getSkillName)

    q_cols = ["attempt_count", "correct", "ms_first_response", "skill_id"]
    q_cols = ["attempt_count", "correct", "ms_first_response"]

    questions = data.groupby("problem_id")[q_cols]

    funcs_to_apply = {c: 'mean' for c in q_cols}
    # funcs_to_apply["skill_id"] = "max"

    questions = questions.agg(funcs_to_apply)
    normalised = normalise_df(questions)
    questions[q_cols[:-1]] = normalised[q_cols[:-1]]
    questions = normalised
    questions.reset_index(level=0, inplace=True)

    u_cols = ["attempt_count", "correct", "ms_first_response"]
    users = data.groupby("user_id")[u_cols].mean()
    users = normalise_df(users)
    users.reset_index(level=0, inplace=True)
    return data, users, questions


def get_edge_pairs(a_questions_ids):
    perms = permutations(a_questions_ids, 2)
    return [p for p in perms if p[0] <= p[1]]


def gen_graph(data, users, questions, add_edges = True):

    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    # user_data = users.to_dict('records')
    # nodes = [(data['user_id'], data) for data in user_data]
    # B.add_nodes_from(nodes, bipartite=0)

    question_data = questions.to_dict('records')
    nodes = [(data['problem_id'], data) for data in question_data]
    B.add_nodes_from(nodes, bipartite=1)

    if add_edges:
        all_assignments = data["assignment_id"].to_list()
        all_users = data["user_id"].to_list()
        zip_assignment_user = zip(all_assignments, all_users)

        for a,u in set(zip_assignment_user):
            condition = (data["assignment_id"] == a) #& (data["user_id"] == u)
            a_questions_ids = data[condition]["problem_id"].to_list()
            edges = get_edge_pairs(a_questions_ids)
            B.add_edges_from(edges)




    print("Nodes: ",len(B.nodes))
    print("Edges: ",len(B.edges))

    return B

def getSkillID(ids):
    if isinstance(ids, int) or isinstance(ids, float):
        return ids

    ids = ids.split(",")
    return ids[0]

def getSkillName(ids):

    ids = ids.split(",")
    if ids[0] == "" or ids[0] == " ":
        return "Unknown"

    return ids[0]


if __name__ == '__main__':

    dataset = AssignmentDataset(isTraining=True,training_data=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    end = len(dataset)
    i = 0
    print(end)
    for d in dataloader:
        print(d)

        i += 1
        if i > 20:
            break