import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse



c_map = {1: (1,0,0),2:(0,1,0), 3:(0,0.6,1), 0:(1,1,0), 4:(1,0,1)}

users = ["A", "B", "C", "D"]
qs = 40
questions = np.arange(0,qs)
questions_skills = np.array([(i // 10) for i in range(qs)])

G = nx.Graph()
G.add_nodes_from(questions, bipartite = 1, questions_skills = questions_skills)

edges = [("A",2),("A",6),("A",9),("A",11),("A",14),("A",15),
         ("C",30),("C",21),("C",25),("C",11),("C",14),("C",18),
         ("B",2),("B",6),("B",0),("B",36),("B",35),("B",32),
         ("D",24),("D",21),("D",25),("D",36),("D",31),("D",32)]



colours = np.concatenate((questions_skills, np.ones(4) * 4), axis = None)
colours = list(map(c_map.get, colours))
pos = nx.circular_layout(G)
G.add_nodes_from(users, bipartite = 0)
G.add_edges_from(edges)
pos["A"] = np.array([0.2,0.2])
pos["B"] = np.array([0.2,-0.2])
pos["C"] = np.array([-0.2,0.2])
pos["D"] = np.array([-0.2,-0.2])
nx.draw(G, pos = pos, node_color = colours, with_labels=True)

plt.show()
negative = 1 - nx.adjacency_matrix(G).todense() - np.eye(44)
neg_graph = G.copy()
neg_graph.remove_edges_from(edges)
all_neg_u, all_neg_v = np.where(negative != 0)
neg_edges = list(zip(all_neg_u, all_neg_v))
neg_graph.add_edges_from(neg_edges)
pos[40] = np.array([0.2,0.2])
pos[41] = np.array([0.2,-0.2])
pos[42] = np.array([-0.2,0.2])
pos[43] = np.array([-0.2,-0.2])
nx.draw(neg_graph, pos = pos, node_color = colours, with_labels=True)

plt.show()