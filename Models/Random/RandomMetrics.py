import pandas as pd
import random

from prettytable import PrettyTable
from tqdm import trange

from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from Models.MetricBase import MetricBase
from Models.URW.URW import UnweightedRandomWalk
from Models.URW.URW_Metrics import URW_Metrics
from datareader import Datareader
from helper_funcs import MRR, AveragePrecision, Recall


class RandomChoiceMetrics(MetricBase):

    def __init__(self, dataset):

        self.items = dataset.item_ids
        # self.metadata = pd.DataFrame(self.metadata, columns=["problem_id", "skill_id", "skill_name"])
        super(RandomChoiceMetrics, self).__init__()

    def top_n_questions(self, anchor, search_size):
        # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])

        if search_size <= len(self.items):
            return self.items.sample(search_size).to_list()
        else:
            metadata_size = len(self.items)
            return self.items.sample(metadata_size).to_list() + [0] * (search_size - metadata_size)

    def rank_questions(self, ids, anchor):
        random.shuffle(ids)
        return ids



def test_model(datareader):

    metrics = []
    datasets = []


    for d in [datareader.train, datareader.test]:
        d = TestDataset(d)
        metric = RandomChoiceMetrics(d)
        metrics.append(metric)
        datasets.append(d)

    model_names = ["Train", "Test"]

    params = zip(metrics, datasets, model_names)

    search_size = 100
    ap_length = 20
    tests = 5000
    # samples = 1000
    output = PrettyTable()
    output.field_names = ["Data"] + model_names

    recalls = []
    rec_ranks = []
    average_precisions = []

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
            ap = AveragePrecision(positive_ids, top_n)
            top_n = metric.top_n_questions(anchor, total_interactions-1)
            rec = Recall(positive_ids, top_n)


            metric.mrr_ranks.append(mrr)
            metric.average_precisions.append(ap)
            metric.recall.append(rec)

        # mr = metric.mean_rank()
        metric_mrr = metric.mean_reciprocal_rank()
        metric_ap = metric.get_average_precision()
        metric_rec = metric.get_recall()
        # ranks.append(mr)
        rec_ranks.append(metric_mrr)
        average_precisions.append(metric_ap)
        recalls.append(metric_rec)

    output.add_row(["Mean Reciprocal Rank"] + [str(r) for r in rec_ranks])
    output.add_row(["Average Precision"] + [str(r) for r in average_precisions])
    output.add_row(["Recall"] + [str(r) for r in recalls])
    print(output)

    return output


if __name__ == '__main__':
    file = "../../skill_builder_data.csv"
    train_reader = Datareader(file, size=00000, training_frac=0.75)

    out = test_model(train_reader)



