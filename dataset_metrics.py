import pandas as pd

from Datasets.Training import TrainDataset
from datareader import Datareader
import networkx as nx
import matplotlib.pyplot as plt

# filename = "skill_builder_data.csv"
# datareader = Datareader(filename, training_frac = 1)
#
#
# dataset = TrainDataset(datareader.interactions)
#
# user_df = dataset.users_df.reset_index()
# questions_df = dataset.items_df.reset_index()
# interactions = dataset.interaction_df
#
#
# user_counts = interactions.groupby("user_id").size()
# question_counts = interactions.groupby("problem_id").size()
# skill_counts = questions_df.groupby("skill_id").size()
#
# def data_analysis(series, name):
#     print("Answers Per {} mean: {}".format(name, series.mean()))
#     print("Answers Per {} median: {}".format(name, series.median()))
#     print("Answers Per {} std: {}".format(name, series.std()))
#     print("Max # of {} Answers : {}".format(name, series.max()))
#     print("Min # of {} Answers : {}".format(name, series.min()))
#
# print("Users:", len(user_df))
# print("Questions:", len(questions_df))
# print("Questions Answered:", len(interactions))
# print("Skills:", len(skill_counts))
# print()
# data_analysis(user_counts, "user")
# print()
# data_analysis(question_counts, "questions")
# print()
# print("Non Skilled_Questions:", skill_counts[0])
# print("Skilled_Questions:", len(questions_df) - skill_counts[0])
#
# interactions_ex = interactions.head(1).squeeze()
# questions_ex = questions_df.head(1).squeeze()
# user_ex = user_df.head(1).squeeze()
#
# xs = []
# ys = []
# counts = user_counts.unique().tolist()
# counts.sort()
#
# for c in counts:
#     xs.append(c)
#     c_df = user_counts.loc[user_counts == c]
#     ys.append(len(c_df))
#
# import matplotlib.pyplot as plt
# plt.hist(xs)
# plt.show()
# import numpy as np
# from scipy.sparse import coo
#
# G = nx.Graph()
# G.add_nodes_from([0,1,2,3,4])
# edges = [(1,2),(2,3),(3,4),(0,1), (4,0)]
# G.add_edges_from(edges)
#
# fig, axs = plt.subplots(1,2)
#
# nx.draw(G, pos=nx.spring_layout(G), with_labels=True, ax = axs[0])
# neg_G = nx.Graph()
# neg_G.add_nodes_from([0,1,2,3,4])
# adj = 1 - nx.adjacency_matrix(G).todense() - np.eye(5)
# u,v = np.where(adj != 0)
# edges = list(zip(u,v))
#
# neg_G.add_edges_from(edges)
# nx.draw(neg_G, pos=nx.spring_layout(neg_G), with_labels=True, ax = axs[1])
# plt.show()
