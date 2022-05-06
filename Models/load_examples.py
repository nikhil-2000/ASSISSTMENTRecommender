from prettytable import PrettyTable
from torch.utils.data import DataLoader
import pickle

from Datasets.GraphDataset import GraphDataset
from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from Models.GCN.LinkPredictorMetrics import LinkPredictorMetrics
from Models.NeuralNetwork.NeuralNetworkMetrics import NormalNNMetrics
from Models.URW.URW import UnweightedRandomWalk
from Models.URW.URW_Metrics import URW_Metrics
from datareader import Datareader
from helper_funcs import AveragePrecision, Recall
import pandas as pd


def create_NN_metric(reader):
    # testWeightsFolder(datareader)
    model_file = "512_10_0.1_0.005_20000_May04_21-00-17_FINAL.pth"
    model_file_path = "NeuralNetwork/WeightFiles/" + model_file
    data = TestDataset(reader.train)
    loader = DataLoader(data, batch_size=1024)
    train_metric = NormalNNMetrics(loader, model_file_path, data)

    data = TestDataset(reader.test)
    loader = DataLoader(data, batch_size=1024)
    metric = NormalNNMetrics(loader, model_file_path, data, (train_metric.embeddings, train_metric.metadata))
    filename = "NN_metric.pickle"
    filehandler = open(filename, 'wb')
    pickle.dump(metric, filehandler)


def create_GCN_metric(reader):
    _in_embs = 50

    all_data = TrainDataset(reader.interactions)
    user_ids = all_data.users_df.index.unique().tolist()
    item_ids = all_data.item_ids.unique().tolist()

    graphDataset = GraphDataset(user_ids, item_ids, reader.train, reader.validation, reader.test)

    model_file = "350_0.01_50_100_100May05_15-54-07.pth"
    # testDataset = TrainDataset(test_reader.ratings_df, test_reader.user_df, test_reader.items_df)
    model_path = "GCN/WeightFiles/"
    test_metric = LinkPredictorMetrics(model_path + model_file, graphDataset, graphDataset.train_graph)

    filename = "GCN_test_metric.pickle"
    filehandler = open(filename, 'wb')
    pickle.dump(test_metric, filehandler)


def create_URW_metric(reader):
    train_data = TrainDataset(reader.train)
    urw = UnweightedRandomWalk(train_data)

    urw.closest = 10

    test_data = TestDataset(reader.test)
    metric = URW_Metrics(test_data, urw)

    filename = "URW_test_metric.pickle"
    filehandler = open(filename, 'wb')
    pickle.dump(metric, filehandler)


def load_NN_metric():
    filename = "NN_metric.pickle"
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)


def load_GCN_metric():
    filename = "GCN_test_metric.pickle"
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)


def load_URW_metric():
    filename = "URW_test_metric.pickle"
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)


def id_to_skill_info(dataset: TestDataset, ids):
    interactions = dataset.interaction_df
    skill_ids = []
    skill_names = []
    for _id in ids:
        skill_interactions = interactions.loc[interactions.problem_id == _id]
        row = skill_interactions.iloc[0]
        skill_ids.append(row.skill_id)
        skill_names.append(row.skill_name)

    return skill_ids, skill_names


def run():
    # create_GCN_metric(train_reader)
    # create_NN_metric(train_reader)
    # create_URW_metric(train_reader)

    neuralNetworkMetric: NormalNNMetrics = load_NN_metric()
    print("1")
    GCNMetric: LinkPredictorMetrics = load_GCN_metric()
    print("2")
    PESMetric: URW_Metrics = load_URW_metric()
    print("3")

    testDataset = PESMetric.dataset
    search_size = 10

    for i in range(100):
        total_interactions = 0
        while total_interactions < search_size or total_interactions > search_size + 10:
            user = testDataset.sample_user()
            total_interactions = len(user.interactions)

            anchor = user.interactions.iloc[0]
            actual_ids = user.interactions.problem_id.tolist()
            nn_top_n = neuralNetworkMetric.top_n_questions(anchor, search_size)
            gcn_top_n = GCNMetric.top_n_items_edges(anchor, search_size)
            pes_top_n = PESMetric.top_n_questions(anchor, search_size)

            if Recall(actual_ids, gcn_top_n) == 0 or max(Recall(actual_ids, nn_top_n), Recall(actual_ids,pes_top_n)) == 0:
                total_interactions = 0

        print("GCN_precision:", AveragePrecision(actual_ids, gcn_top_n), Recall(actual_ids, gcn_top_n))
        print("NN_precision:", AveragePrecision(actual_ids, nn_top_n), Recall(actual_ids, nn_top_n))
        print("PES_precision:", AveragePrecision(actual_ids, pes_top_n), Recall(actual_ids, pes_top_n))

        results = [actual_ids, gcn_top_n, nn_top_n, pes_top_n]
        # skill_ids, skill_names = [],[]
        # for rs in results:
        #     s_ids, s_names= id_to_skill_info(testDataset, rs)
        #     skill_ids.append(s_ids)
        #     skill_names.append(s_names)

        # 4 x 10
        # 4 x 10
        # 4 x 10

        # all_data = [res + skill_ids[i] + skill_names[i] for i, res in enumerate(results)]

        table = PrettyTable()
        table.field_names = ["Actual",
                             "GCN - ID",  # "GCN - Skill ID", "GCN - Skill Name",
                             "NN - ID",  # NN - Skill ID", "NN - Skill Name",
                             "PES - ID", ]  # "PES - Skill ID", "PES - Skill Name"]
        names = list(zip(*results))
        for row in names:
            table.add_row(row)

        print(table)

    # Pick random user
    # Get top-n for them
    # Transfer IDs to name
    # Put it all in a single dataframe


def skill_check():
    _ids = "53669	85478	54078	53603	53665	54001	84920	54031	53671	53672"
    _ids = [int(n) for n in _ids.split("\t")]


    gcn_ids = "54031	61092	61116	85909	88605	54055	87507	49278	97860	85906"
    gcn_ids = [int(n) for n in gcn_ids.split("\t")]

    nn_ids = "49380	53709	53737	53653	49365	48484	49358	85486	53665	49331"
    nn_ids = [int(n) for n in nn_ids.split("\t")]

    pes_ids = "85951	53984	49275	53781	54087	89741	53992	53991	85303	54012"
    pes_ids = [int(n) for n in pes_ids.split("\t")]

    train_reader = Datareader("../skill_builder_data.csv", size=0, training_frac=0.75, val_frac=0.25)

    all = [_ids,gcn_ids, nn_ids, pes_ids]
    for each in all:

        df = train_reader.interactions
        df = df.loc[df.problem_id.isin(each)]
        df = df.drop_duplicates("problem_id")[["problem_id", "skill_id", "skill_name"]]
        for _id in each:
            row = df.loc[df.problem_id == _id]
            print(row.skill_id.item(),"|", row.skill_name.item())

        print()


    print(set(gcn_ids).intersection(_ids))
    print(set(nn_ids).intersection(_ids))
    print(set(pes_ids).intersection(_ids))

if __name__ == '__main__':
    skill_check()