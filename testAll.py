import math
import random

from torch.utils.data import DataLoader
from tqdm import trange



from prettytable import PrettyTable
import numpy as np

from Datasets.Testing import TestDataset
from Models.NeuralNetwork.NeuralNetworkMetrics import NormalNNMetrics
from Models.Random.RandomMetrics import RandomChoiceMetrics
from Models.URW.URW import UnweightedRandomWalk
from Models.URW.URW_Metrics import URW_Metrics
from datareader import Datareader


def run_metrics(model, samples = 1000, tests = 1000, size = 0):
    datareader = Datareader("skill_builder_data.csv", size=0, training_frac=0.7)
    train = TestDataset(datareader.train)
    test = TestDataset(datareader.test)

    train_loader = DataLoader(train)
    test_loader = DataLoader(test)

    datasets = {"train": train, "test":test}
    dataloaders = {"train": train_loader, "test":test_loader}
    phases = ["train", "test"]

    urw = UnweightedRandomWalk(train)

    for phase in phases:
        dataset, dataloader = datasets[phase] , dataloaders[phase]

        neural_network_metrics = NormalNNMetrics(dataloader, model, dataset)
        urw_metrics = URW_Metrics(dataset, urw)
        random_metrics = RandomChoiceMetrics(dataset)


    # models = [urw_metrics]
        metrics = [urw_metrics, neural_network_metrics, random_metrics]

        search_size = 30


    # model_names = ["Random Walk", "Same Skill", "Random Choice", "Collab Filtering"]
        model_names = ["URW", "Neural Network", "Random"]
        print("\nTests Begin")
        for i in trange(tests):
            # Pick Random User
            total_interactions = 0
            while total_interactions < 5:
                user = urw_metrics.sample_user()
                user_interactions, total_interactions = user.interactions, len(user.interactions)
            # Generate Anchor Positive
            a_idx, p_idx = random.sample(range(0, total_interactions), 2)
            anchor = user_interactions.iloc[a_idx]
            anchor_id = anchor.problem_id.item()

            positive = user_interactions.iloc[p_idx]
            positive_id = positive.problem_id.item()

            without_positive = dataset.item_ids[~dataset.item_ids.isin(user_interactions.problem_id.unique())]
            random_ids = np.random.choice(without_positive,samples).tolist()
            all_ids = random_ids + [positive_id]
            random.shuffle(all_ids)

            # Find n Closest
            for i,m in enumerate(metrics):
                # top_n = m.top_n_questions(anchor, search_size)
                ranking = m.rank_questions(all_ids, anchor)

                # set_prediction = set(top_n)
                # if positive_id in set_prediction:
                #     m.hits += 1

                rank = ranking.index(positive_id)

                m.ranks.append(rank)

            # else:
            #     print(same_skill_metrics.ranks[-1])


        output = PrettyTable()
        output.field_names = ["Phase", "Model", "Mean Rank"]
        for i, name in enumerate(model_names):
            m = metrics[i]

            # hr = m.hitrate(tests)
            mr = m.mean_rank()
            output.add_row([phase, name, mr])


        print(output)



if __name__ == '__main__':

    model_file = "D:\My Docs/University\Year 4\Individual Project/Assisstments_Dataset/Models/NeuralNetwork/WeightFiles/64_30_0.05_0.01.pth"
    run_metrics(model_file, samples = 1000, tests = 1000, size = 0)

    # run_metrics("../non_skill_builder_data_new.csv", "questions_dataset_Mar10_16-48-43.pth", samples = 500, tests = 1000, size = 0000)

