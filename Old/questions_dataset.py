import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import random
import math

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class QuestionDataset(Dataset):

    def __init__(self, data_file, is_training = True, size = 0):
        self.interactions, self.users, self.questions =  self.get_data(data_file, subset_data = size)

        self.is_training = is_training
        self.gen_graph()


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        current_question = self.questions.loc[index]["problem_id"]

        # random walk from current question
        dict_counter = dict()
        dict_counter[current_question] = 1

        # Traversing through the neighbors of start node
        for i in range(1000):
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

        del dict_counter[current_question]
        counts = list(dict_counter.items())
        by_visits = sorted(counts, key=lambda x: x[1], reverse=True)
        positives = by_visits[:10]
        negatives = by_visits[10:50]

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

    def get_data(self, data_file, subset_data = 0):
        data = pd.read_csv(data_file, encoding='latin-1')

        if subset_data:
            data = data.head(subset_data)
        data['skill_id'] = data['skill_id'].fillna(0)
        data["skill_id"] = data["skill_id"].apply(getSkillID)

        data['skill_name'] = data['skill_name'].fillna("Unknown")
        data["skill_name"] = data["skill_name"].apply(getSkillName)

        # q_cols = ["attempt_count", "correct", "ms_first_response", "skill_id"]
        q_cols = ["attempt_count", "correct", "ms_first_response"]

        questions = data.groupby("problem_id")[q_cols]

        funcs_to_apply = {c: 'mean' for c in q_cols}
        # funcs_to_apply["skill_id"] = "max"

        questions = questions.agg(funcs_to_apply)
        normalised = normalise_df(questions)
        # problems[q_cols[:-1]] = normalised[q_cols[:-1]]
        questions = normalised
        questions.reset_index(level=0, inplace=True)

        u_cols = ["attempt_count", "correct", "ms_first_response"]
        users = data.groupby("user_id")[u_cols].mean()
        users = normalise_df(users)
        users.reset_index(level=0, inplace=True)
        return data, users, questions

    def gen_graph(self):

        B = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        user_data = self.users.to_dict('records')
        nodes = [(data['user_id'], data) for data in user_data]
        B.add_nodes_from(nodes, bipartite=0)

        question_data = self.questions.to_dict('records')
        nodes = [(data['problem_id'], data) for data in question_data]
        B.add_nodes_from(nodes, bipartite=1)

        if self.is_training:
            for i, row in self.interactions.iterrows():
                user, question = row["user_id"], row["problem_id"]
                B.add_edge(user, question)

        print("Nodes: ", len(B.nodes))
        print("Edges: ", len(B.edges))

        self.graph = B


class EmbeddingDataset(QuestionDataset):

    def __init__(self, data_file, is_training=False, subset_data_size = 0):
        super(EmbeddingDataset, self).__init__(data_file, is_training= is_training, size= subset_data_size)
        self.interactions["skill_id"] = self.interactions["skill_id"].apply(lambda x : int(x))


    def __getitem__(self, index):
        q_id = self.questions.loc[index]["problem_id"]
        skill_id, skill_name = self.map_id_to_name(q_id)

        features = self.get_features(q_id)
        return features,skill_id, skill_name, q_id

    def map_id_to_name(self,q_id):
        condition = self.interactions["problem_id"] == q_id
        df = self.interactions.loc[condition][["skill_name","skill_id"]]
        potential_names_ids = list(df.itertuples(index = False, name = None))
        if not potential_names_ids:
            name = "Unknown"
            skill_id = 0
        else:
            name,skill_id = random.choice(potential_names_ids)

        return name, skill_id



def normalise_df(df):
    return (df - df.min()) / (df.max() - df.min())

def getSkillID(ids):
    if isinstance(ids, int) or isinstance(ids, float):
        return ids

    ids = ids.split(",")
    return int(ids[0])

def getSkillName(ids):

    ids = ids.split(",")
    if ids[0] == "" or ids[0] == " ":
        return "Unknown"

    return ids[0]

def test_training_set():
    file = "../skill_builder_data.csv"
    dataset = QuestionDataset(file, is_training=True, size = 1000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    end = len(dataset)
    i = 0
    for d in tqdm(dataloader):
        # print(d)
        pass
        # i += 1
        # if i > 20:
        #     break

def test_embeddings_set():
    file = "../skill_builder_data.csv"
    dataset = EmbeddingDataset(file, is_training=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    end = len(dataset)
    i = 0
    for d in tqdm(dataloader):
        # print(d)
        pass
        # i += 1
        # if i > 20:
        #     break


if __name__ == '__main__':
    test_embeddings_set()