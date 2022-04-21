import torch
import numpy as np
from tqdm import tqdm

from Models.NeuralNetwork.NeuralNetworkModel import EmbeddingNetwork


class CalcEmbeddings:

    def __init__(self, dataloader, model_file):
        self.model_file = model_file
        self.set_model()
        self.dataloader = dataloader


    def set_model(self):
        weight_path = self.model_file
        loaded_file = torch.load(weight_path)
        state_dict = loaded_file["model_state_dict"]
        n_skills = loaded_file["n_skills"]
        model = EmbeddingNetwork(n_skills = n_skills)


        model.load_state_dict(state_dict)

        self.model = model

    def get_embeddings(self):
        embeddings = []
        embeddings_dict = {}
        metadata = []
        print("\nGenerating Embeddings")
        count = 0
        for vector, info in tqdm(self.dataloader, leave=True, position=0):
        # for vector, info in self.dataloader:
            skill_ids, skill_names, item_ids = info
            out = self.model(vector, skill_ids)
            out = out.detach().numpy()
            embeddings.extend(out)
            count += len(item_ids)
            item_ids = item_ids.detach().tolist()
            skill_ids = skill_ids.detach().tolist()

            metadata.extend(list(zip(skill_ids, skill_names, item_ids)))
            for i, id in enumerate(item_ids):
                embeddings_dict[id] = out[i]

        embeddings = np.array(embeddings).squeeze()

        return embeddings, metadata, embeddings_dict
