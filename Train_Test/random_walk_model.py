# Aim to generate good embeddings for problems
import datetime

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from Objects.NeuralNetworkModel import EmbeddingNetwork
from Objects.TriplesGenerator import TriplesGenerator
from Old.questions_dataset import *
import numpy as np
import sys

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
    # <batch size> <num epochs> <margin> <output_name>
    argv = argv[1:]
    usagemessage = "Should be 'python random_walk_model.py <train_file_name> <batch size> <num epochs> <margin> <output_name>'"
    if len(argv) < 4:
        print(usagemessage)
        return

    data_file = argv[0]

    batch = int(argv[1])
    assert batch > 0, "Batch size should be more than 0\n" + usagemessage

    numepochs = int(argv[2])
    assert numepochs > 0, "Need more than " + str(numepochs) + " epochs\n" + usagemessage

    outpath = argv[4] + "_" + datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    margin = float(argv[3])
    assert 0 < margin, "Pick a margin greater than 0\n" + usagemessage

    # phases = ["train"]

    print('Triplet embeddings training session. Inputs: ' + str(
        batch) + ', ' + str(numepochs) + ', ' + str(margin) + ', ' + outpath)
    #
    # print("Validation will happen ? ", doValidation)

    # train_ds = QuestionDataset()
    train_ds = TriplesGenerator(data_file, True, size = 0)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)

    # Allow all parameters to be fit
    model = EmbeddingNetwork(4)

    # model = torch.jit.script(model).to(device) # send model to GPU
    isParallel = torch.cuda.is_available()
    if isParallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = model.to(device)  # send model to GPU

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.jit.script(TripletLoss(margin=10.0))
    criterion = TripletLoss(margin=margin)

    # let invalid epochs pass through without training
    if numepochs < 1:
        numepochs = 0
        loss = 0

    run_name = datetime.datetime.now().strftime("%b%d_%H-%M-%S") + "_Epochs" + str(numepochs) + "_Datasize" + str(
        len(train_ds))
    writer = SummaryWriter(log_dir="runs/" + run_name)

    train_steps = 0

    for epoch in tqdm(range(numepochs), desc="Epochs"):
        # Split data into "Batches" and calc distances

        dataset, data_loader = train_ds, train_loader

        losses = []
        for step, (anchor_question, positive_question, negative_question) in enumerate(
                tqdm(data_loader, leave=True, position=0)):
            anchor_question = anchor_question.to(device)  # send tensor to GPU
            positive_question = positive_question.to(device)  # send tensor to GPU
            negative_question = negative_question.to(device)  # send tensor to GPU

            anchor_out = model(anchor_question)
            positive_out = model(positive_question)
            negative_out = model(negative_question)
            # Clears space on GPU I think
            del anchor_question
            del positive_question
            del negative_question
            # Triplet Loss !!! + Backprop
            loss = criterion(anchor_out, positive_out, negative_out)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())

            # batch_norm = torch.linalg.norm(anchor_out, ord = 1, dim= 1)
            # embedding_norm = torch.mean(batch_norm)
            # writer.add_scalar("Loss/embedding_norm", embedding_norm, s)

            writer.add_scalar("triplet_loss", loss, train_steps)

            batch_positive_loss = torch.mean(criterion.calc_euclidean(anchor_out, positive_out))
            batch_negative_loss = torch.mean(criterion.calc_euclidean(anchor_out, negative_out))
            writer.add_scalar("Other/Positive_Loss", batch_positive_loss, train_steps)
            writer.add_scalar("Other/Negative_Loss", batch_negative_loss, train_steps)
            writer.add_scalar("Pos_Neg_Difference", batch_negative_loss - batch_positive_loss,
                              train_steps)

            train_steps += batch

        writer.add_scalar("Epoch_triplet_loss", np.mean(losses), epoch + 1)

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, numepochs, np.mean(losses)))

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
    learn(sys.argv)

# Annealing on the margin
# Experiment with hyperparameters
