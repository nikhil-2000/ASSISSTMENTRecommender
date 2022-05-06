import os
import pickle
import random
from datetime import datetime

from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from Datasets.Testing import TestDataset
from Models.NeuralNetwork.NeuralNetworkMetrics import NormalNNMetrics
from Models.NeuralNetwork.visualise_embeddings import add_embeddings_to_tensorboard
from Models.URW.URW import UnweightedRandomWalk
from datareader import Datareader
from helper_funcs import MRR, Recall, AveragePrecision


def test_model(datareader, model_file):

    # Allow all parameters to be fit

    metrics = []
    datasets = []
    dataloaders = []

    # for d in [datareader.train, datareader.validation]:
    filename = "../NN_metric.pickle"
    filehandler = open(filename, 'rb')
    train_metric =  pickle.load(filehandler)
    # metrics.append(train_metric)
    # datasets.append(data)
    # dataloaders.append(loader)
    #
    # data = TestDataset(datareader.validation)
    # loader = DataLoader(data, batch_size=1024)
    # metric = NormalNNMetrics(loader, model_file, data, (train_metric.embeddings, train_metric.metadata))
    # metrics.append(metric)
    # datasets.append(data)
    # dataloaders.append(loader)

    data = TestDataset(datareader.test)
    loader = DataLoader(data, batch_size=1024)
    metric = NormalNNMetrics(loader, model_file, data, (train_metric.embeddings, train_metric.metadata))
    metrics.append(metric)
    datasets.append(data)
    dataloaders.append(loader)

    # model_names = ["Train","Validation", "Test"]
    model_names = ["Test"]

    # params = zip(metrics, datasets, dataloaders, model_names)

    params = zip(metrics, datasets, model_names)

    results = []

    search_size = 100
    ap_length = 20
    tests = 1000

    for metric, data, name in params:

        print("\nTesting " + name)
        for i in trange(tests):
            # for i in range(tests):
            # Pick Random User
            total_interactions = 0
            while total_interactions < 5:
                user = data.sample_user()
                user_interactions, total_interactions = user.interactions, len(user.interactions)
            # Generate Anchor Positive
            a_idx, p_idx = random.sample(range(0, total_interactions), 2)
            anchor = user_interactions.iloc[a_idx]

            positive_ids = data.item_ids[data.item_ids.isin(user_interactions.problem_id.unique())]



            positive_ids = positive_ids[data.item_ids != anchor.problem_id.item()]

            #random.shuffle(all_samples)


            top_n = metric.top_n_questions(anchor, search_size)
            mrr = MRR(positive_ids, top_n)
            top_n = metric.top_n_questions(anchor, ap_length)
            ap = AveragePrecision(positive_ids, top_n)
            top_n = metric.top_n_questions(anchor, total_interactions - 1)
            rec = Recall(positive_ids, top_n)


            metric.mrr_ranks.append(mrr)
            metric.average_precisions.append(ap)
            metric.recall.append(rec)

        # mr = metric.mean_rank()
        metric_mrr = metric.mean_reciprocal_rank()
        metric_ap = metric.get_average_precision()
        metric_rec = metric.get_recall()
        # ranks.append(mr)

        results.append([metric_mrr, metric_ap, metric_rec])

    return results


def visualise(datareader, model_file, name):
    name = name + datetime.now().strftime("%b%d_%H-%M-%S")
    add_embeddings_to_tensorboard(datareader, model_file,name)


def testWeightsFolder(datareader):

    train_table = PrettyTable()
    train_table.field_names = ["Model","Mean Reciprocal Rank","Average Precision","Recall By User"]
    validation_table = PrettyTable()
    validation_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    test_table = PrettyTable()
    test_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    for model_file in tqdm(os.listdir("WeightFiles")):
        model_file_path = "WeightFiles/" + model_file
        train_row, val_row, test_row = test_model(datareader, model_file_path)
        train_table.add_row([model_file] + [str(r) for r in train_row])
        validation_table.add_row([model_file] + [str(r) for r in val_row])
        test_table.add_row([model_file] + [str(r) for r in test_row])
        print(train_table)
        print()
        print(validation_table)
        print()
        print(test_table)
        with open("results-mean-std.txt", "w") as f:
            f.write(str(train_table) + "\n" + str(validation_table) + "\n" + str(test_table))

        visualise(datareader,model_file_path, "Embeddings")





if __name__ == '__main__':
    filename = "../../skill_builder_data.csv"

    tables = []
    model_files = []
    rank_table = PrettyTable()
    rank_table.field_names = ["Model", "Train", "Val", "Test"]

    hr_table = PrettyTable()
    hr_table.field_names = ["Model", "Train", "Val", "Test"]
    datareader = Datareader(filename, size=0, training_frac=0.75, val_frac=0.25)
    # testWeightsFolder(datareader)
    model_file = "512_10_0.1_0.005_20000_May04_21-00-17_FINAL.pth"
    model_file_path = "WeightFiles/" + model_file

    test = test_model(datareader, model_file_path)
    # print(train)
    # print(val)
    print(test)
    # visualise(datareader, model_file_path, model_file)