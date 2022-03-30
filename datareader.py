import pandas as pd
from tqdm import tqdm
import numpy as np

class Datareader:

    def __init__(self, filename, size=0, training_frac = 0.7):
        dataset = pd.read_csv(filename, encoding="latin-1")
        if size > 0:
            dataset = dataset.head(size)

        dataset['skill_id'] = dataset['skill_id'].fillna(0)
        dataset["skill_id"] = dataset["skill_id"].apply(getSkillID)
        dataset["skill_id"] = dataset["skill_id"].apply(int)
        dataset["skill_name"] = dataset["skill_name"].fillna("Unknown")
        dataset = dataset.sample(frac=1)
        train_length = int(len(dataset) * training_frac)
        test_length = len(dataset) - train_length
        self.interactions = dataset
        self.train = dataset.head(train_length)
        self.test = dataset.tail(test_length)
        # assert len(self.interactions) + len(self.test_data) == len(dataset)




def getSkillID(ids):
    if isinstance(ids, int) or isinstance(ids, float):
        return ids

    ids = ids.split(",")
    return ids[0]

