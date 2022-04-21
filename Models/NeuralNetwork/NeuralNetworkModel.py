import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=50, n_skills=None):
        super().__init__()
        emb_output = 10
        skill_output = 20
        self.fc1 = nn.Linear(in_channels, emb_output)
        self.fc2 = nn.Linear(emb_output, emb_output)

        self.embeddings = torch.nn.Embedding(n_skills, skill_output)
        self.lin = nn.Linear(emb_output + skill_output , out_channels)


    def forward(self, x, s):
        # s is a 1d long tensor mapping to skill IDs
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.relu(x)

        e = self.embeddings(s)

        z = self.lin(torch.cat([x, e], dim=1))

        return z
