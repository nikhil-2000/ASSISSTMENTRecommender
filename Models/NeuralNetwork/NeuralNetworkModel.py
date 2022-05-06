import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class EmbeddingNetwork(nn.Module):
    def __init__(self, in_channels=6, out_channels=20, n_skills=None):
        super().__init__()
        emb_output = 20
        skill_output = 20
        self.fc1 = nn.Linear(in_channels, 10)
        self.fc2 = nn.Linear(10, emb_output)

        self.embeddings = torch.nn.Embedding(n_skills, skill_output)
        self.lin = nn.Linear(emb_output + skill_output , out_channels)


    def forward(self, x, s):
        # s is a 1d long tensor mapping to skill IDs
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        e = self.embeddings(s)

        if len(e.shape) == 1:
            combined = torch.cat([x, e])
        else:
            combined = torch.cat([x, e], dim=1)

        z = self.lin(combined)

        return z

if __name__ == '__main__':


    writer = SummaryWriter("runs/model")
    model = EmbeddingNetwork(n_skills=380)
    inputs = torch.Tensor([[1,1,1,1,1,1],[1,1,1,1,1,1]])
    skill = torch.Tensor([0,1]).long().squeeze()
    writer.add_graph(model, input_to_model=[inputs, skill])
    writer.flush()