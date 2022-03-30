# Aim to generate good embeddings for problems
import datetime
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Datasets.Training import TrainDataset
from Models.NeuralNetwork.NeuralNetworkModel import EmbeddingNetwork
import numpy as np
import sys
from pytorch_metric_learning import miners, losses

from datareader import Datareader

"""
Hide categories and see results
triplet loss with problems as the 'pins'
Positive = Do a random walk on the graph starting from the question, pick a positive from the 5 most visited
Negative = a random question that the user hasn't answered

"""
"""
Have user data, question data, + interaction data as seperate table
Question data builds question nodes
Same for user data
Interaction data will create edges on the graph
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = torch.nn.functional.cosine_similarity(anchor, positive)
        distance_negative_anchor = torch.nn.functional.cosine_similarity(anchor, negative)
        # distance_negative_positive = self.calc_euclidean(positive, negative)
        # Test Cosine similarity
        losses = torch.relu(distance_positive - distance_negative_anchor + self.margin)

        return losses.mean()


def learn(argv):
    usagemessage = "Should be 'python train_NN.py <batch size> <num epochs> <margin> <output_name>'"
    if len(argv) < 4:
        print(usagemessage)
        return

    batch = int(argv[0])
    assert batch > 0, "Batch size should be more than 0\n" + usagemessage

    numepochs = int(argv[1])
    assert numepochs > 0, "Need more than " + str(numepochs) + " epochs\n" + usagemessage

    margin = float(argv[2])
    assert 0 < margin, "Pick a margin greater than 0\n" + usagemessage

    lr = float(argv[3])

    outpath = argv[4]

    filename = argv[5]

    print('Triplet embeddings training session. Inputs: ' + str(
        batch) + ', ' + str(numepochs) + ', ' + str(margin) + ', ' + outpath)
    #
    # print("Validation will happen ? ", doValidation)

    datareader = Datareader(filename, size=0, training_frac=0.7)

    train_ds = TrainDataset(datareader.train)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)

    # Allow all parameters to be fit
    model = EmbeddingNetwork(3)

    # model = torch.jit.script(model).to(device) # send model to GPU
    isParallel = torch.cuda.is_available()
    if isParallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = model.to(device)  # send model to GPU

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = torch.jit.script(TripletLoss(margin=10.0))

    easy = miners.TripletMarginMiner(margin=margin, type_of_triplets="all")
    semi_hard = miners.TripletMarginMiner(margin=margin, type_of_triplets="semi-hard")
    hard = miners.TripletMarginMiner(margin=margin, type_of_triplets="hard")
    loss_func = losses.TripletMarginLoss(margin=margin)

    # let invalid epochs pass through without training
    if numepochs < 1:
        numepochs = 0
        loss = 0

    run_name = outpath
    writer = SummaryWriter(log_dir="runs/" + run_name)

    train_steps = 0

    for epoch in tqdm(range(numepochs), desc="Epochs"):
        # Split data into "Batches" and calc distances

        dataset, data_loader = train_ds, train_loader

        epoch_losses = []
        for step, (features, labels) in enumerate(
                tqdm(data_loader, leave=True, position=0)):
            optimizer.zero_grad()
            features = features.to(device)  # send tensor to GPU

            embeddings = model(features)
            # Clears space on GPU I think
            del features

            # Triplet Loss !!! + Backprop
            percent = epoch / numepochs
            if percent < 0.10:
                pairs = easy(embeddings, labels)
            elif 0.10 <= percent <= 0.30:
                pairs = semi_hard(embeddings, labels)
            else:
                pairs = hard(embeddings, labels)

            loss = loss_func(embeddings, labels, pairs)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.cpu().detach().numpy())

            writer.add_scalar("triplet_loss", loss, train_steps)

            train_steps += batch

        writer.add_scalar("Epoch_triplet_loss", np.mean(epoch_losses), epoch + 1)

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, numepochs, np.mean(epoch_losses)))
        # Saves model so that distances can be updated using new model

        weights = model.module.state_dict() if isParallel else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': weights,
            'optimzier_state_dict': optimizer.state_dict(),
            'loss': loss
        }, outpath + '.pth')

        dataset.modelfile = outpath + '.pth'


if __name__ == '__main__':
    batches = [64, 128, 256]
    epochs = [30]
    margins = [0.05, 0.1, 0.5]
    lrs = [0.0001, 0.001, 0.01, 1]
    filename = "../../skill_builder_data.csv"

    params = [batches, epochs, margins, lrs]

    for ps in product(*params):
        b, e, m, l = ps
        s = [str(x) for x in ps]
        output = "WeightFiles/" + "_".join(s)
        learn([b, e , m ,l , output, filename])
# Annealing on the margin
# Experiment with hyperparameters
