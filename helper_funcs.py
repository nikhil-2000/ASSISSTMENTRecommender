import numpy as np
from tqdm import tqdm
# vector_features = ["attempt_count", "correct", "ms_first_response","skill_id_normalised"]
vector_features = ["attempt_count", "correct", "ms_first_response"]
metadata_cols = ["(User,Question)","User ID","Problem_id","Skill ID","Skill Name", "User Attempts","User Correct", "User Response", "Question Attempts", "Question Correct","Question Response"]


def normalise(df):
    return (df-df.min())/(df.max()-df.min())
    # return (df-df.mean())/(df.std())


def add_metrics(interactions):
    q_cols = ["attempt_count", "correct", "ms_first_response", "skill_id"]
    questions = interactions.groupby(["problem_id","user_id"])[q_cols]

    funcs_to_apply = {c: 'mean' for c in q_cols}
    funcs_to_apply["skill_id"] = "max"

    questions = questions.agg(funcs_to_apply)
    normalised = normalise(questions)
    questions[q_cols[:-1]] = normalised[q_cols[:-1]]
    questions["skill_id_normalised"] = normalise(questions["skill_id"])

    # problems.reset_index(level=0, inplace=True)

    u_cols = ["attempt_count", "correct", "ms_first_response"]
    users = interactions.groupby("user_id")[u_cols].mean()
    users = normalise(users)

    return questions, users

def MRR(positive_ids, top_n_rec):

    for i, rec_id in enumerate(top_n_rec):
        if rec_id in positive_ids:
            return 1/(i+1)

    return 0

def AveragePrecision(positive_ids, top_n_rec):
    precision_at_i = []
    positives = 0
    total = 0
    for i, rec_id in enumerate(top_n_rec):
        total += 1
        if rec_id in positive_ids:
            positives += 1

        precision_at_i.append(positives / total)

    return np.mean(precision_at_i)

def Recall(positive_ids, top_n_rec):
    total = len(positive_ids)
    rec = sum([1 for rec_id in top_n_rec if rec_id in positive_ids])
    return rec/total


def convert_distances(distances, search_size):
    x = np.exp(-distances)
    weights = x / x.sum(axis=0)
    x = search_size * weights
    x = np.round(x)
    leftover = search_size - sum(x)
    if leftover > 0:
        i = 0
        while leftover > 0:
            x[i] += 1
            leftover -= 1
            i = (i+1) % search_size
    elif leftover < 0:
        i = len(x) - 1
        while leftover < 0:
            x[i] -= 1
            leftover += 1
            i = (i-1) % search_size


    return x