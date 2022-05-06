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

        for vector, skill_ids, info, keys in tqdm(self.dataloader, leave=True, position=0):
            keys = torch.stack(keys).detach().T.tolist()

            rows_without_skill_name = info[:3] + info[4:]
            rows_without_skill_name = torch.stack(rows_without_skill_name).detach().tolist()
            rows_without_skill_name.insert(3,list(info[3]))
            rows_without_skill_name.insert(0, keys)

            out = self.model(vector, skill_ids)
            out = out.detach().numpy()
            embeddings.extend(out)
            # count += len(item_ids)
            rows = list(tuple(zip(*rows_without_skill_name)))
            metadata.extend(rows)
            keys = map(tuple, keys)
            key_vals = list(zip(keys, out))
            # for i, k in enumerate(keys):
            #     embeddings_dict[k] = out[i]
            embeddings_dict.update(key_vals)



        embeddings = np.array(embeddings).squeeze()

        return embeddings, metadata, embeddings_dict
