import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from Objects.ScoreFolder import ScoreFolder
from Train_Test.random_walk_model import EmbeddingNetwork


class CalcEmbeddings:

    def __init__(self, data_file, model_file, size = 0):
        self.model_file = model_file
        self.set_model()
        self.dataset = ScoreFolder(data_file, size = size)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)


    def set_model(self):
        model = EmbeddingNetwork()

        weight_path = self.model_file
        state_dict = torch.load(weight_path)["model_state_dict"]
        model.load_state_dict(state_dict)

        self.model = model

    def get_embeddings(self):
        embeddings = []
        embeddings_dict = {}
        metadata = []
        for input, skill_name, skill_id, q_id in tqdm(self.dataloader):
            out = self.model(input)
            out = out.detach().numpy()
            embeddings.append(out)

            q_id = int(q_id.detach().numpy())
            skill_id = int(skill_id.detach().numpy())
            skill_name = skill_name[0]
            row = (q_id, skill_id, skill_name)
            metadata.append(row)
            embeddings_dict[q_id] = out


        embeddings = np.array(embeddings).squeeze()

        return embeddings, metadata, embeddings_dict
