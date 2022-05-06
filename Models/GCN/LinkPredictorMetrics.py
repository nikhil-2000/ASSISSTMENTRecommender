import torch



import numpy as np

from Datasets.GraphDataset import GraphDataset
from Models.GCN.LinkPredictor import DotPredictor, GraphSAGE, MLPPredictor
from Models.MetricBase import MetricBase


class LinkPredictorMetrics(MetricBase):
    

    def __init__(self ,model_file, graphDataset: GraphDataset, train_graph):
        super(LinkPredictorMetrics, self).__init__()

        self.model_file = model_file
        self.dataset = graphDataset
        self.graph = train_graph

        self.user_edges, self.question_edges = train_graph.edges(etype = "attempts")
        self.user_edges = self.user_edges.detach().numpy().squeeze()
        self.question_edges = self.question_edges.detach().numpy().squeeze()
        self.train_graph = train_graph

        self.set_model()
        self.add_embeddings_to_graph()
        # self.get_score_edges()



    def set_model(self):
        checkpoint = torch.load(self.model_file)
        in_feats, hidden_feats, emb_size = checkpoint['in_feat'], checkpoint['hidden_feat'], checkpoint['emb_size']
        model = GraphSAGE(in_feats, hidden_feats, emb_size)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.pred = MLPPredictor(emb_size)
        self.pred.load_state_dict(checkpoint['mlp_state_dict'])

        self.model = model

    def add_embeddings_to_graph(self):
        h = self.model(self.train_graph, self.train_graph.ndata['feat'])
        self.graph.ndata["h"] = h

    def top_n_items(self, anchor, search_size):

        user_id = anchor.user_id.item()
        user_id =  self.dataset.user_to_idx[user_id]
        # idx = (self.graph.nodes("user")==user_id).nonzero()
        user_h = self.graph.ndata["h"]["user"][user_id].squeeze()


        embeddings = self.graph.ndata["h"]["question"]
        scores = self.pred.score_user(user_h, embeddings).detach().numpy() #torch.matmul(user_h.T, embeddings.T).tolist()
        # items = self.graph.nodes("question").numpy()

        sorted_idxs = np.argsort(scores)[::-1]
        graph_items = sorted_idxs[:search_size]
        sorted_items = [self.dataset.idx_to_item[i] for i in graph_items]


        return sorted_items


    def top_n_items_edges(self, anchor, search_size):

        user_id = anchor.user_id.item()
        user_id = self.dataset.user_to_idx[user_id]
        # idx = (self.graph.nodes("user")==user_id).nonzero()
        user_h = self.graph.ndata["h"]["user"][user_id].squeeze()

        embeddings = self.graph.ndata["h"]["question"]
        scores = self.pred.score_user(user_h, embeddings).detach().numpy() #torch.matmul(user_h.T, embeddings.T).tolist()
        # items = self.graph.nodes("question").numpy()

        # sorted_idxs = np.argsort(scores)[::-1]
        graph_items = np.argsort(scores)[::-1]


        user_eids = (self.user_edges == user_id).nonzero()[0]
        connected_to = self.question_edges[user_eids]

        sorted_items = list(filter(lambda x: not x in connected_to, graph_items))[:search_size]
        sorted_items = list(map(self.dataset.idx_to_item.get,sorted_items))



        return sorted_items

