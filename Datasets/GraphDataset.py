import dgl
import torch
from dgl.data import DGLDataset
from torch import nn

from Datasets.Training import TrainDataset
from datareader import Datareader
from helper_funcs import normalise
import pandas as pd


class GraphDataset(DGLDataset):

    def __init__(self, user_ids, item_ids, train_interactions, val_interactions, test_interactions, node_embedding=100):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.train = train_interactions
        self.validation = val_interactions
        self.test = test_interactions

        self.user_to_idx = {u: i for i, u in enumerate(user_ids)}
        self.idx_to_user = {i: u for i, u in enumerate(user_ids)}

        self.item_to_idx = {u: i for i, u in enumerate(item_ids)}
        self.idx_to_item = {i: u for i, u in enumerate(item_ids)}

        self.user_count, self.item_count = len(user_ids), len(item_ids)

        torch.manual_seed(1)
        self.user_embeddings_layer = nn.Embedding(self.user_count, node_embedding)
        torch.manual_seed(1)
        self.item_embeddings_layer = nn.Embedding(self.item_count, node_embedding)

        user_nodes = torch.arange(self.user_count).long()
        item_nodes = torch.arange(self.item_count).long()

        self.user_embeddings = self.user_embeddings_layer(user_nodes)
        self.item_embeddings = self.item_embeddings_layer(item_nodes)

        super(GraphDataset, self).__init__("Assistment")

    def process(self):
        self.train_graph = self.buildGraph(self.train)
        self.validation_graph = self.buildGraph(self.validation)
        self.test_graph = self.buildGraph(self.test)

    def buildGraph(self, interactions):
        user_edge = list(map(self.user_to_idx.get, interactions.user_id.tolist()))
        item_edge = list(map(self.item_to_idx.get, interactions.problem_id.tolist()))


        graph = dgl.heterograph({
            ("user", "attempts", "question"): (user_edge, item_edge)
            , ("question", "attempted_by", "user"): (item_edge, user_edge)
        }, num_nodes_dict={"user": self.user_count, "question": self.item_count})

        graph.nodes["user"].data["feat"] = self.user_embeddings
        graph.nodes["question"].data["feat"] = self.item_embeddings

        return graph


if __name__ == '__main__':
    file = "../skill_builder_data.csv"
    reader = Datareader(file, size=10000, training_frac=0.7, val_frac=0.3)
    all_data = TrainDataset(reader.interactions)
    user_ids = all_data.users_df.index.unique().tolist()
    item_ids = all_data.item_ids.unique().tolist()

    g_data = GraphDataset(user_ids, item_ids, reader.train, reader.validation, reader.test)

    print()
