import os
import random
from datetime import datetime

from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from Datasets.Testing import TestDataset
from Models.MetricBase import MetricBase
from Models.NeuralNetwork.compute_embeddings import CalcEmbeddings
from helper_funcs import metadata_cols
import numpy as np
import pandas as pd
import random


class NormalNNMetrics(MetricBase):

    def __init__(self, dataloader, model_file, dataset, train_embeddings_metadata = None):
        self.embedder = CalcEmbeddings(dataloader, model_file)
        self.dataset = dataset
        self.generated_embeddings = False

        if train_embeddings_metadata == None:
            self.embeddings, self.metadata, self.e_dict = self.embedder.get_embeddings()
            self.train_embeddings = self.embeddings
            self.metadata = pd.DataFrame(self.metadata, columns=metadata_cols)
            self.generated_embeddings = True

        else:
            embs, metadata = train_embeddings_metadata
            self.train_embeddings = embs
            self.metadata = pd.DataFrame(metadata, columns=metadata_cols)

        super(NormalNNMetrics, self).__init__()

    def top_n_questions(self, anchor, search_size):
        # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])
        key = (anchor.problem_id.item(), anchor.user_id.item())
        if not self.generated_embeddings:
            vector, skill_ids = self.dataset.get_by_ids(key[1],key[0])
            anchor_embedding = self.embedder.model(vector,skill_ids).detach().numpy()
        else:
            anchor_embedding = self.e_dict[key]
        # anchor_embedding = self.e_dict[key]

        dists = cosine_dists(anchor_embedding, self.train_embeddings) #np.linalg.norm(self.embeddings - anchor_embedding, axis=1)
        sorted_indexes = np.argsort(dists)
        # best_indexes = sorted_indexes[1:search_size + 1]
        sorted_ids = self.metadata.iloc[sorted_indexes].Problem_id.drop_duplicates().tolist()

        if len(sorted_ids) >= search_size:
            return sorted_ids[:search_size]
        else:
            return sorted_ids + [0] * (search_size - len(sorted_ids))

    def rank_questions(self, all_interactions, anchor):
        key = (anchor.problem_id.item(), anchor.user_id.item())
        anchor_embedding = self.e_dict[key]

        embeddings_to_rank = []
        ids_found = []
        ids_not_found = []
        for i, interaction in all_interactions.iterrows():
            k = (interaction.problem_id, interaction.user_id)
            embedding = self.e_dict.get(k, np.array([]))
            if embedding.shape[0] > 0:
                embeddings_to_rank.append(embedding)
                ids_found.append(k)
            else:
                ids_not_found.append(k)



        embeddings_to_rank = np.array(embeddings_to_rank).squeeze()

        dists = np.linalg.norm(embeddings_to_rank - anchor_embedding, axis=1)
        sorted_indexes = np.argsort(dists)
        sorted_ids = all_interactions.iloc[sorted_indexes].problem_id.drop_duplicates().tolist()


        return sorted_ids + ids_not_found


def cosine_dists(u, vs):
    u_dot_v = np.sum(u * vs, axis=1)

    # find the norm of u and each row of v
    mod_u = np.sqrt(np.sum(u * u))
    mod_v = np.sqrt(np.sum(vs * vs, axis=1))

    # just apply the definition
    final = 1 - u_dot_v / (mod_u * mod_v)
    return final
