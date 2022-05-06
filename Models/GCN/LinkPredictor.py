
import dgl.nn.pytorch as dglnn
import torch
from torch import nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import apply_each


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, embedding_size):
        super(GraphSAGE, self).__init__()
        #Creates embeddings for size h_feat
        # self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        # self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.conv1 = dglnn.HeteroGraphConv({
            "attempts": dglnn.SAGEConv(in_feats,h_feats, 'mean')
            ,"attempted_by": dglnn.SAGEConv(in_feats,h_feats, 'mean')
        })

        self.conv2 = dglnn.HeteroGraphConv({
            "attempts": dglnn.SAGEConv(h_feats,embedding_size, 'mean')
            ,"attempted_by": dglnn.SAGEConv(h_feats,embedding_size, 'mean')
        })

        self.activation = F.relu

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = apply_each(h,self.activation)
        h = self.conv2(g, h)
        return h


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            #Gets embeddings for all nodes
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            # Apply edges will do this pairwise on all existing edges in the graph
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = "attempts")

            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][('user', 'attempts', 'question')][:, 0]

    def score_user(self, user_embedding, item_embeddings):

        len_items = item_embeddings.shape[0]
        user_matrix = user_embedding.repeat(len_items, 1)
        h = torch.cat([user_matrix, item_embeddings], 1)
        return torch.matmul(user_embedding.T, item_embeddings.T)


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, h_feats // 2)
        self.W3 = nn.Linear(h_feats // 2, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W3(F.relu(self.W2(F.relu(self.W1(h))))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges, etype = "attempts")
            return g.edata['score'][('user', 'attempts', 'question')]

    def score_user(self, user_embedding, item_embeddings):

        len_items = item_embeddings.shape[0]
        user_matrix = user_embedding.repeat(len_items, 1)
        h = torch.cat([user_matrix, item_embeddings], 1)
        return self.W3(F.relu(self.W2(F.relu(self.W1(h))))).squeeze(1)