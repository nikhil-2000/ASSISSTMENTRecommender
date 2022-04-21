import os
import random
from datetime import datetime

from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from Datasets.Testing import TestDataset
from Models.NeuralNetwork.compute_embeddings import CalcEmbeddings
from Models.NeuralNetwork.visualise_embeddings import add_embeddings_to_tensorboard
from datareader import Datareader
import numpy as np
import pandas as pd
import random


class NormalNNMetrics:

    def __init__(self, dataloader, model_file, dataset):
        self.embedder = CalcEmbeddings(dataloader, model_file)
        self.dataset = dataset

        self.embeddings, self.metadata, self.e_dict = self.embedder.get_embeddings()
        self.metadata = pd.Series(self.metadata)
        self.users = self.create_users()

        self.hits = 0
        self.ranks = []

    def sample_user(self):
        return random.choice(self.users)

    def create_users(self):
        interactions = self.dataset.interaction_df

        user_ratings = interactions.sort_values("user_id")
        all_users = user_ratings.user_id.unique()
        all_items = self.dataset.item_ids
        users = []
        # print(len(all_skills))
        # skill_vectors = pd.DataFrame(columns=["user_id"] + list(all_skills))
        # print("\nCreating Users")
        # for user in tqdm(all_users):
        for user in all_users:
            u_ratings = user_ratings[user_ratings["user_id"] == user]
            new_user = User(u_ratings)
            new_user.generate_q_completed_df(all_items)
            users.append(new_user)

        return users

    def top_n_questions(self, anchor, search_size):
        # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])
        anchor_id = anchor.problem_id.item()
        anchor_embedding = self.e_dict[anchor_id]
        dists = np.linalg.norm(self.embeddings - anchor_embedding, axis=1)
        sorted_indexes = np.argsort(dists)
        best_indexes = sorted_indexes[1:search_size + 1]

        if len(best_indexes) >= search_size:
            return self.metadata.iloc[best_indexes].to_list()
        else:
            return self.metadata.iloc[best_indexes].to_list() + [0] * (search_size - len(best_indexes))

    def rank_questions(self, ids, anchor):
        anchor_id = anchor.problem_id.item()
        anchor = self.e_dict[anchor_id]

        embeddings_to_rank = []
        ids_found = []
        ids_not_found = []
        for k in ids:
            embedding = self.e_dict.get(k, np.array([]))
            if embedding.shape[0] > 0:
                embeddings_to_rank.append(embedding)
                ids_found.append(k)
            else:
                ids_not_found.append(k)



        embeddings_to_rank = np.array(embeddings_to_rank).squeeze()

        dists = np.linalg.norm(embeddings_to_rank - anchor, axis=1)
        sorted_indexes = np.argsort(dists)

        return np.array(ids_found)[sorted_indexes].tolist() + ids_not_found

    def hitrate(self, tests):
        return 100 * self.hits / tests

    def mean_rank(self):
        return sum(self.ranks) / len(self.ranks)


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


def test_model(datareader, model_file):

    # Allow all parameters to be fit

    metrics = []
    datasets = []
    dataloaders = []

    for d in [datareader.train, datareader.validation, datareader.test]:
        data = TestDataset(d)
        loader = DataLoader(data, batch_size=64)
        metric = NormalNNMetrics(loader, model_file, data)
        metrics.append(metric)
        datasets.append(data)
        dataloaders.append(loader)

    model_names = ["Train", "Val" ,"Test"]

    params = zip(metrics, datasets, dataloaders, model_names)

    search_size = 100
    tests = 1000
    samples = 1000
    output = PrettyTable()
    output.field_names = ["Data", "Hitrate", "Mean Rank"]

    ranks = []
    hitrates = []
    for metric, data, loader, name in params:

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
            if any([pos in set_prediction for pos in user_interactions.problem_id]):
                metric.hits += 1

            rank = ranking.index(positive_id)

            metric.ranks.append(rank)

        hr = metric.hitrate(tests)
        mr = metric.mean_rank()
        output.add_row([name, hr, mr])
        ranks.append(mr)
        hitrates.append(hr)

    return output, ranks, hitrates

def visualise(datareader, model_file, name):
    name = name + datetime.now().strftime("%b%d_%H-%M-%S")
    add_embeddings_to_tensorboard(datareader, model_file,name)


def testWeightsFolder(datareader):

    model_files = []
    rank_table = PrettyTable()
    rank_table.field_names = ["Model", "Train", "Val", "Test"]

    hr_table = PrettyTable()
    hr_table.field_names = ["Model", "Train", "Val", "Test"]

    for model_file in tqdm(os.listdir("WeightFiles")):
        model_file_path = "WeightFiles/" + model_file
        t, ranks, hitrates = test_model(datareader, model_file_path)
        model_files.append(model_file)
        rank_table.add_row([model_file] + [str(r) for r in ranks])
        hr_table.add_row([model_file] + [str(h) for h in hitrates])

        print(rank_table)
        print(hr_table)
        visualise(datareader,model_file_path, "Embeddings" )



if __name__ == '__main__':
    file = "../../skill_builder_data.csv"

    tables = []
    model_files = []
    rank_table = PrettyTable()
    rank_table.field_names = ["Model", "Train", "Val", "Test"]

    hr_table = PrettyTable()
    hr_table.field_names = ["Model", "Train", "Val", "Test"]
    datareader = Datareader(file, size=0, training_frac=0.7, val_frac=0.2)
    testWeightsFolder(datareader)