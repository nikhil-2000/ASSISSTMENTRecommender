import pandas as pd
from tqdm import tqdm
import numpy as np

class Dataset:

    def __init__(self, filename, size=0):
        dataset = pd.read_csv(filename, encoding="latin-1")
        if size > 0:
            dataset = dataset.head(size)

        dataset['skill_id'] = dataset['skill_id'].fillna(0)
        dataset["skill_id"] = dataset["skill_id"].apply(getSkillID)
        dataset["skill_id"] = dataset["skill_id"].apply(int)
        self.interactions = dataset

        q_cols = ["attempt_count", "correct", "ms_first_response","skill_id"]
        questions = dataset.groupby("problem_id")[q_cols]

        funcs_to_apply = {c: 'mean' for c in q_cols}
        funcs_to_apply["skill_id"] = "max"

        questions = questions.agg(funcs_to_apply)
        normalised = normalise_df(questions)
        # problems[q_cols[:-1]] = normalised[q_cols[:-1]]
        questions = normalised
        # problems.reset_index(level=0, inplace=True)

        u_cols = ["attempt_count", "correct", "ms_first_response"]
        users = dataset.groupby("user_id")[u_cols].mean()
        users = normalise_df(users)
        # users.reset_index(level=0, inplace=True)
        self.problems = questions
        self.users = users

        self.problem_ids = pd.Series(self.problems.index.to_series())
        self.user_ids = pd.Series(self.users.index.to_series())



    def __len__(self):
        return len(self.problem_ids)



def getSkillID(ids):
    if isinstance(ids, int) or isinstance(ids, float):
        return ids

    ids = ids.split(",")
    return ids[0]

def normalise_df(df):
    return (df - df.min()) / (df.max() - df.min())

