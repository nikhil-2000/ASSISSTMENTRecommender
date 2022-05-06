import random
import os

from prettytable import PrettyTable
from tqdm import trange

from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from Models.URW.URW import UnweightedRandomWalk
from Models.URW.URW_Metrics import URW_Metrics
from datareader import Datareader
from helper_funcs import MRR, AveragePrecision, Recall


def test_model(train_reader, urw):

    metrics = []
    datasets = []


    #
    # for d in [train_reader.train, train_reader.test]:
    d = TestDataset(train_reader.test)
    metric = URW_Metrics(d,urw)
    metrics.append(metric)
    datasets.append(d)

    model_names = ["Test"]

    params = zip(metrics, datasets, model_names)

    results = []

    search_size = 100
    ap_length = 20
    tests = 1000
    # samples = 1000

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


            top_n = metric.top_n_questions(anchor, search_size)
            mrr = MRR(positive_ids, top_n)
            top_n = metric.top_n_questions(anchor, ap_length)
            ap = AveragePrecision(positive_ids, top_n[:ap_length])
            top_n = metric.top_n_questions(anchor, total_interactions-1)
            rec = Recall(positive_ids, top_n[:total_interactions-1])


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

def test_main_model(closest = 10):
    file = "../../skill_builder_data.csv"
    train_reader = Datareader(file, size=0, training_frac=0.75, val_frac=0.25)

    data = TrainDataset(train_reader.train)
    train_table = PrettyTable()
    train_table.field_names = ["k", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    test_table = PrettyTable()
    test_table.field_names = ["k", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    urw = UnweightedRandomWalk(data)

    urw.closest = closest
    test_row = test_model(train_reader, urw)[0]
    print(test_row)
    # train_table.add_row([urw.closest] + [str(r) for r in train_row])
    # test_table.add_row([urw.closest] + [str(r) for r in test_row])
    # print(train_table)
    # print()
    # print(test_table)
    # with open("results.txt", "w") as f:
    #     f.write(str(train_table) + "\n" + str(test_table) + "\n")



def hyperparams():
    file = "../../skill_builder_data.csv"
    train_reader = Datareader(file, size=0, training_frac=0.75, val_frac=0.25)

    data = TrainDataset(train_reader.train)
    c = [5,10,20,30,40,50,75,100]
    train_table = PrettyTable()
    train_table.field_names = ["k","Mean Reciprocal Rank","Average Precision","Recall By User"]


    validation_table = PrettyTable()
    validation_table.field_names = ["k", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    test_table = PrettyTable()
    test_table.field_names = ["k", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    urw = UnweightedRandomWalk(data)

    for n in c:
        urw.closest = n
        # train_row, test_row = test_model(train_reader, urw)
        validation_row = test_model(train_reader, urw)[0]
        validation_table.add_row([urw.closest] + [str(r) for r in validation_row])
        # train_table.add_row([urw.closest] + [str(r) for r in train_row])
        # test_table.add_row([urw.closest] + [str(r) for r in test_row])
        # print(train_table)
        # print()
        # print(test_table)
        with open("validation.txt", "w") as f:
            f.write(str(validation_table))
    # print(output)

if __name__ == '__main__':
    test_main_model(5)
