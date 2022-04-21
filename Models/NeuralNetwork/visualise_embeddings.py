from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from Datasets.Testing import TestDataset
from Models.NeuralNetwork.compute_embeddings import CalcEmbeddings
from datareader import Datareader
from torch.utils.data import DataLoader


def write_data_to_board(xs, run_name, metadata=[], headers=[]):
    writer = SummaryWriter(log_dir="runs/" + run_name)
    if metadata == [] or headers == "":
        writer.add_embedding(xs)
    else:
        writer.add_embedding(xs, metadata=metadata, metadata_header = headers)
    writer.flush()

def add_embeddings_to_tensorboard(datareader,model_file, file_name):
    train = TestDataset(datareader.interactions)
    loader = DataLoader(train, batch_size=64)


    train_embedder = CalcEmbeddings(loader, model_file)
    embeddings, metadata, _ = train_embedder.get_embeddings()

    print("# of Embeddings : ", len(embeddings))
    run_name = datetime.now().strftime("%b%d_%H-%M-%S") + "_" + file_name

    write_data_to_board(embeddings,run_name, metadata=metadata, headers = ["Skill ID","Skill Name", "Q_ID"])


if __name__ == '__main__':
    model_file = "WeightFiles/128_30_0.05_0.001_added_skill_embedding_Apr20_22-58-34.pth"
    add_embeddings_to_tensorboard("../../skill_builder_data.csv", model_file, "training_data")
