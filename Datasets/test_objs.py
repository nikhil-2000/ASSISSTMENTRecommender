from torch.utils.data import DataLoader

from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from datareader import Datareader

datareader = Datareader("../skill_builder_data.csv", size=0)
train = TrainDataset(datareader.train)
test = TestDataset(datareader.test)

train_loader = DataLoader(train)
test_loader = DataLoader(test)

