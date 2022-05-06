import torch
from pytorch_metric_learning import miners, distances, losses
from tqdm import tqdm

from Datasets.Training import TrainDataset
from Models.NeuralNetwork.NeuralNetworkModel import EmbeddingNetwork
from datareader import Datareader
import numpy as np
import pandas as pd

batch = 1024
numepochs = 1

margin = 0.5

device = torch.device("cpu")


# print('Triplet embeddings training session. Inputs: ' + str(
#     batch) + ', ' + str(numepochs) + ', ' + str(margin) + ', ' + outpath)
# #
# print("Validation will happen ? ", doValidation)

datareader = Datareader("../../skill_builder_data.csv", size = 10000, training_frac=1, val_frac=0.25)
train_ds = TrainDataset(datareader.train)
# train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
skill_count = 380  # datareader.interactions.skill_id.max() + 1

val_ds = TrainDataset(datareader.validation)
# val_loader = DataLoader(val_ds, batch_size=batch, num_workers=0)
# Allow all parameters to be fit
model = EmbeddingNetwork(in_channels=6, n_skills=skill_count)

# model = torch.jit.script(model).to(device) # send model to GPU

model = model.to(device)  # send model to GPU

# criterion = torch.jit.script(TripletLoss(margin=10.0))
all_triplets = miners.TripletMarginMiner(margin=margin, type_of_triplets="all",
                                         distance=distances.CosineSimilarity())
loss_func = losses.TripletMarginLoss(margin=margin)

# easy = easy.to(device)
# semi_hard = semi_hard.to(device)
# hard = hard.to(device)
# let invalid epochs pass through without training
if numepochs < 1:
    numepochs = 0
    loss = 0

# train_writer = SummaryWriter(log_dir="runs/" + run_name)
# val_writer = SummaryWriter(log_dir="runs/" + run_name + "_val")

train_steps = 0
val_steps = 0
steps = 0

samples = 10000

for epoch in tqdm(range(numepochs), desc="Epochs"):
    # Split data into "Batches" and calc distances
    train_subset = torch.utils.data.Subset(train_ds, np.random.choice(len(train_ds), samples))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1024, shuffle=True)

    val_subset = torch.utils.data.Subset(val_ds, np.random.choice(len(val_ds), samples))
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1024, shuffle=True)
    dataset, data_loader = train_ds, train_loader
    #
    # if percent <= 0.30:
    #     miner = semi_hard
    # else:
    #     miner = hard

    miner = all_triplets

    for phase in ["train", "validation"]:
        epoch_losses = []

        if phase == "train":
            model.train()
            loader = train_loader
            # writer = train_writer
            val_steps = steps
            steps = train_steps
        else:
            model.eval()
            loader = val_loader
            # writer = val_writer
            train_steps = steps
            steps = val_steps

        for step, (features, skill_ids, user_ids) in enumerate(
                tqdm(loader, leave=True, position=0)):

            # if phase == "train": optimizer.zero_grad()
            df = pd.DataFrame(columns = ["Vector","Skill ID","User ID"])
            features = features.to(device)  # send tensor to GPU

            embeddings = model(features, skill_ids)
            # Clears space on GPU I think
            pairs = miner(embeddings, user_ids)
            a,p,n = torch.stack(pairs)[:, 700].squeeze()
            df.loc["Anchor"] = [features[a].tolist(), skill_ids[a].item(), user_ids[a].item()]
            df.loc["Positive"] = [features[p].tolist(), skill_ids[p].item(), user_ids[p].item()]
            df.loc["Negative"] = [features[n].tolist(), skill_ids[n].item(), user_ids[n].item()]
            # Triplet Loss !!! + Backprop
            loss = loss_func(embeddings, user_ids, pairs)

            if phase == "train":
                loss.backward()
                # optimizer.step()

            if phase == "train":
                steps += batch
            else:
                steps += batch