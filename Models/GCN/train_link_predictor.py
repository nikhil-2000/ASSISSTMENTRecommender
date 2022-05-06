import itertools
from datetime import datetime

import dgl
import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Datasets.GraphDataset import GraphDataset
from Datasets.Training import TrainDataset
from Models.GCN.LinkPredictor import GraphSAGE, DotPredictor, MLPPredictor
from Models.GCN.run_eval import test_model
from datareader import Datareader
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from itertools import product

import dgl.function as fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_loss(pos_score, neg_score):
    # Pos score a vector with positive edge scores i.e scores of edges which exist
    # Neg score a vector with negative edge scores i.e scores of edges which don't exist
    scores = torch.cat([pos_score, neg_score])
    # First edges are labeled 1 as they are positive and
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    labels = labels.to(device)
    # Should push likely edges towards having a score of 1 and unlikely edges have a score of 0
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    # Same as above
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    # Measures difference between labels and actual guesses
    return roc_auc_score(labels, scores)


def split_data(g):
    # Selects edges in (u,v) format and shuffles all

    u, v = g.edges(etype="attempts")
    n_of_rating_edges = g.number_of_edges(etype="attempts")
    eids = np.arange(n_of_rating_edges)
    eids = np.random.permutation(eids)

    # Select 10% of edges as test edges, rest are training
    test_size = int(len(eids) * 0.1)
    train_size = n_of_rating_edges - test_size

    # The positive test edges are the first 10% of (u,v) pairs
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    # The positive training edges are the remaining 90% (u,v) pairs
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Gets adjacency matrix for graph
    adj = g.adj(etype="attempts").to_dense().numpy()
    # Inverts adjacency matrix, all edges that didn't exists, now do except self loops
    adj_neg = 1 - adj  # - np.eye(g.number_of_nodes())
    # Select (u,v) pairs of all edges that don't exist
    neg_u, neg_v = np.where(adj_neg != 0)

    # Samples an equivalent number of negative edges that exist in the graph
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges(etype="attempts"))
    # The negative test edges are the first 10% of (u,v) pairs
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    # The negative training edges are the remaining 90% (u,v) pairs
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # Remove the positive test edges, this is what we will test the model on and predict if they exist
    train_g = dgl.remove_edges(g, eids[:test_size], etype="attempts")
    train_g = dgl.remove_edges(train_g, eids[:test_size], etype="attempted_by")

    positive_train = (train_pos_u, train_pos_v)
    positive_test = (test_pos_u, test_pos_v)
    negative_train = (train_neg_u, train_neg_v)
    negative_test = (test_neg_u, test_neg_v)

    all_negs = (neg_u, neg_v)

    return train_g, positive_train, positive_test, negative_train, negative_test, all_negs


def pos_neg_graphs(graph):
    pos_u, pos_v = graph.edges(etype="attempts")
    n_of_rating_edges = graph.number_of_edges(etype="attempts")
    eids = np.arange(n_of_rating_edges)

    all_neg_u, all_neg_v = get_negatives(graph)

    # Samples an equivalent number of negative edges that exist in the graph
    neg_eids = np.random.choice(len(all_neg_u), graph.number_of_edges(etype="attempts"))
    neg_u, neg_v = all_neg_u[neg_eids], all_neg_v[neg_eids]

    return (pos_u, pos_v), (neg_u, neg_v)


def get_negatives(graph):
    adj = graph.adj(etype="attempts").to_dense().numpy()
    # Inverts adjacency matrix, all edges that didn't exists, now do except self loops
    adj_neg = 1 - adj  # - np.eye(g.number_of_nodes())
    # Select (u,v) pairs of all edges that don't exist
    all_neg_u, all_neg_v = np.where(adj_neg != 0)
    return all_neg_u, all_neg_v


def construct_graph(graph, edges):
    u, v = edges
    new_graph = dgl.heterograph({("user", "attempts", "question"): (u, v),
                                 ("question", "attempted_by", "user"): (v, u)},
                                num_nodes_dict={"user": graph.number_of_nodes("user"),
                                                "question": graph.number_of_nodes("question")})

    return new_graph


def learn(train_graphs, val_graphs, test_graphs, args):
    # train_graph, train_pos, train_neg = train_graphs
    # val_graph, val_pos, val_neg = val_graphs
    run_name = "_".join([str(a) for a in args]) + datetime.now().strftime("%b%d_%H-%M-%S")
    numepochs, lr, in_feat, hidden_feats, embedding_size = args

    train_graphs = [g.to(device) for g in train_graphs]
    val_graphs = [g.to(device) for g in val_graphs]
    test_graphs = [g.to(device) for g in test_graphs]

    train_graph, _, _ = train_graphs
    in_feat = train_graph.ndata['feat']["user"].shape[1]
    model = GraphSAGE(in_feat, hidden_feats, embedding_size)
    pred = MLPPredictor(embedding_size)

    model = model.to(device)  # send model to GPU
    pred = pred.to(device)


    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=lr)

    train_writer = SummaryWriter(log_dir="runs/" + run_name + "_train")
    val_writer = SummaryWriter(log_dir="runs/" + run_name + "_val")

    outpaths_auc = []
    epoch = max(numepochs)


    model.train()
    for e in range(epoch+1):
        # forward the node embeddings
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
                g, pos, neg = train_graphs
                writer = train_writer
            else:
                model.eval()
                g, pos, neg = val_graphs
                writer = val_writer

            h = model(g, g.ndata['feat'])
            # Adds score to each edge which represents a probability of the edge existing in both graphs
            pos_score = pred(pos, h)
            neg_score = pred(neg, h)
            loss = compute_loss(pos_score, neg_score)

            # backward
            if phase == "train":
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                if e % 20 == 0:
                    print('In epoch {}, loss: {}'.format(e, loss))





            writer.add_scalar("Epoch_loss/", loss, e + 1)

        if e in numepochs:
            with torch.no_grad():
                h = model(train_graph, train_graph.ndata['feat'])
                train_pos_score = pred(train_graphs[1], h).cpu()
                train_neg_score = pred(train_graphs[2], h).cpu()
                val_pos_score = pred(val_graphs[1], h).cpu()
                val_neg_score = pred(val_graphs[2], h).cpu()
                test_pos_score = pred(test_graphs[1], h).cpu()
                test_neg_score = pred(test_graphs[2], h).cpu()
                print('AUC', compute_auc(val_pos_score, val_neg_score))


            model_args = [e] + args[1:]
            outpath = "WeightFiles/" + "_".join([str(a) for a in model_args]) + datetime.now().strftime("%b%d_%H-%M-%S")

            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimzier_state_dict': optimizer.state_dict(),
                'mlp_state_dict': pred.state_dict(),
                'loss': loss,
                'in_feat': in_feat,
                'hidden_feat': hidden_feats,
                'emb_size': embedding_size
            }, outpath + '.pth')

            train_score = compute_auc(train_pos_score, train_neg_score)
            val_score = compute_auc(val_pos_score, val_neg_score)
            test_score = compute_auc(test_pos_score, test_neg_score)

            model_data =(outpath + '.pth', (train_score, val_score, test_score))
            outpaths_auc.append(model_data)

    return outpaths_auc


def hyperparams():

    initial_embeddings = [50, 100, 200]
    hidden_features = [100,200]
    # output_embeddings = [50, 100]
    # model_feats = [(50,50,200), (50,100,200),(100,50,200),(100,100,200),(100,200,100)]
    lrs = [0.01, 0.005]
    epochs = [200,350,500]

    train_table = PrettyTable()
    train_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    val_table = PrettyTable()
    val_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    test_table = PrettyTable()
    test_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    auc_table = PrettyTable()
    auc_table.field_names = ["Model", "Train", "Validation", "Test"]

    params = [initial_embeddings, hidden_features, lrs]

    file = "../../skill_builder_data.csv"
    reader = Datareader(file, size=0, training_frac=0.75, val_frac=0.25)
    all_data = TrainDataset(reader.interactions)
    trainDataset = TrainDataset(reader.train)
    validationDataset = TrainDataset(reader.validation)
    testDataset = TrainDataset(reader.test)
    user_ids = all_data.users_df.index.unique().tolist()
    item_ids = all_data.item_ids.unique().tolist()

    for run_parameters in product(*params):
        # _epoch, _lr ,_in_embs, _hidden, _out = run_paramters
        _in_embs, _hidden, _lr = run_parameters

        # if _lr == 0.001:
        #     _epoch = 500


        dataset = GraphDataset(user_ids, item_ids, reader.train, reader.validation, reader.test)

        positive_train, negative_train = pos_neg_graphs(dataset.train_graph)
        positive_validation, negative_validation = pos_neg_graphs(dataset.validation_graph)
        positive_test, negative_test = pos_neg_graphs(dataset.test_graph)

        train_pos_g = construct_graph(dataset.train_graph, positive_train)
        # Builds negative graph using interactions that don't exist
        train_neg_g = construct_graph(dataset.train_graph, negative_train)

        val_pos_g = construct_graph(dataset.validation_graph, positive_validation)
        # Builds negative graph using interactions that don't exist
        val_neg_g = construct_graph(dataset.validation_graph, negative_validation)

        test_pos_g = construct_graph(dataset.test_graph, positive_test)
        test_neg_g = construct_graph(dataset.test_graph, negative_test)

        # Same again
        # test_pos_g = construct_graph(train_graph, positive_test)
        # # Builds negative graph using interactions that don't exist
        # test_neg_g = construct_graph(train_graph, negative_test)

        models_data = learn([dataset.train_graph, train_pos_g, train_neg_g],
                                   [dataset.validation_graph, val_pos_g, val_neg_g],
                                   [dataset.test_graph, test_pos_g, test_neg_g],
                                   [epochs, _lr, _in_embs, _hidden, _hidden])

        del positive_train, negative_train
        del positive_validation, negative_validation
        # del positive_test, negative_test
        del train_pos_g
        del train_neg_g
        del val_pos_g
        del val_neg_g

        # Test Dataset With Val Data
        # val_test_interactions = pd.concat([train_reader.validation, test_reader.ratings_df])

        for model_file, scores in models_data:
            val_row, test_row = test_model(model_file, dataset, trainDataset, validationDataset, testDataset)

            # train_table.add_row([model_file] + [str(r) for r in train_row])
            val_table.add_row([model_file] + [str(r) for r in val_row])
            test_table.add_row([model_file] + [str(r) for r in test_row])
            # auc_table.add_row([model_file] + [str(s) for s in scores])

        print(train_table)
        print()
        print(val_table)
        print()
        print(test_table)
        print()
        print(auc_table)
        with open("results-new-final.txt", "w") as f:
            f.write(str(train_table) + "\n" + str(val_table) + "\n" + str(test_table) + "\n" + str(auc_table))

        del dataset

        # print(results)

def single_model():
    train_table = PrettyTable()
    train_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    val_table = PrettyTable()
    val_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    test_table = PrettyTable()
    test_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    auc_table = PrettyTable()
    auc_table.field_names = ["Model", "Train", "Validation", "Test"]


    file = "../../skill_builder_data.csv"
    reader = Datareader(file, size=0, training_frac=0.75, val_frac=0.25)
    all_data = TrainDataset(reader.interactions)
    trainDataset = TrainDataset(reader.train)
    validationDataset = TrainDataset(reader.validation)
    testDataset = TrainDataset(reader.test)
    user_ids = all_data.users_df.index.unique().tolist()
    item_ids = all_data.item_ids.unique().tolist()

    # _epoch, _lr ,_in_embs, _hidden, _out = run_paramters
    _in_embs= 100

    # if _lr == 0.001:
    #     _epoch = 500

    dataset = GraphDataset(user_ids, item_ids, reader.train, reader.validation, reader.test)

    positive_train, negative_train = pos_neg_graphs(dataset.train_graph)
    positive_validation, negative_validation = pos_neg_graphs(dataset.validation_graph)
    positive_test, negative_test = pos_neg_graphs(dataset.test_graph)

    train_pos_g = construct_graph(dataset.train_graph, positive_train)
    # Builds negative graph using interactions that don't exist
    train_neg_g = construct_graph(dataset.train_graph, negative_train)

    val_pos_g = construct_graph(dataset.validation_graph, positive_validation)
    # Builds negative graph using interactions that don't exist
    val_neg_g = construct_graph(dataset.validation_graph, negative_validation)

    test_pos_g = construct_graph(dataset.test_graph, positive_test)
    test_neg_g = construct_graph(dataset.test_graph, negative_test)

    # Same again
    # test_pos_g = construct_graph(train_graph, positive_test)
    # # Builds negative graph using interactions that don't exist
    # test_neg_g = construct_graph(train_graph, negative_test)

    # models_data = learn([dataset.train_graph, train_pos_g, train_neg_g],
    #                     [dataset.validation_graph, val_pos_g, val_neg_g],
    #                     [dataset.test_graph, test_pos_g, test_neg_g],
    #                     [epochs, _lr, _in_embs, _hidden, _hidden])

    del positive_train, negative_train
    del positive_validation, negative_validation
    # del positive_test, negative_test
    del train_pos_g
    del train_neg_g
    del val_pos_g
    del val_neg_g

    # Test Dataset With Val Data
    # val_test_interactions = pd.concat([train_reader.validation, test_reader.ratings_df])
    model_file ="WeightFiles/500_0.005_50_200_200May04_15-37-11.pth"
    results = test_model(model_file, dataset, trainDataset, validationDataset, testDataset)
    print(results)

    # with open("results.txt", "w") as f:
    #     f.write(str(train_table) + "\n" + str(val_table) + "\n" + str(test_table) + "\n" + str(auc_table))

if __name__ == '__main__':
    hyperparams()