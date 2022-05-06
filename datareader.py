import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

class Datareader:

    def __init__(self, filename, size=0, training_frac = 0.7, val_frac = 0):
        dataset = pd.read_csv(filename, encoding="latin-1")
        dataset = dataset.drop_duplicates(subset = ["user_id", "problem_id"])

        if size > 0:
            dataset = dataset.sample(size, random_state = 1)

        dataset['skill_id'] = dataset['skill_id'].fillna(0)
        dataset["skill_id"] = dataset["skill_id"].apply(getSkillID)
        dataset["skill_id"] = dataset["skill_id"].apply(int)
        dataset["skill_name"] = dataset["skill_name"].fillna("Unknown")
        dataset = dataset.sample(frac=1, random_state=1)

        self.interactions = dataset

        if training_frac < 1:
            self.train, self.test = train_test_split(self.interactions, test_size=1 - training_frac, random_state=42)
        else:
            self.train = self.interactions

        if val_frac > 0:
            self.train, self.validation = train_test_split(self.train, test_size=val_frac, random_state=42)

        # assert len(self.interactions) + len(self.test_data) == len(dataset)




def getSkillID(ids):
    if isinstance(ids, int) or isinstance(ids, float):
        return ids

    ids = ids.split(",")
    return ids[0]


if __name__ == '__main__':
    a = Datareader("skill_builder_data.csv",10000,training_frac=0.75, val_frac=0.25)
    b = Datareader("skill_builder_data.csv",10000,training_frac=0.75, val_frac=0.25)

    assert (a.train == b.train).all()[["user_id", "problem_id"]].all()
    assert (a.validation== b.validation).all()[["user_id", "problem_id"]].all()
    assert (a.test== b.test).all()[["user_id", "problem_id"]].all()

