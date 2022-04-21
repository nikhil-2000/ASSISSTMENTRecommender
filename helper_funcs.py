import numpy as np
from tqdm import tqdm
# vector_features = ["attempt_count", "correct", "ms_first_response","skill_id_normalised"]
vector_features = ["attempt_count", "correct", "ms_first_response"]

def normalise(df):
    return (df-df.min())/(df.max()-df.min())


def add_metrics(interactions):
    q_cols = ["attempt_count", "correct", "ms_first_response", "skill_id"]
    questions = interactions.groupby("problem_id")[q_cols]

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

# def add_metrics(interactions, users, items):
#     # A sample can have avg rating, watch count, male viewers, female viewers, most common job, average age
#     item_ids = []
#     attempt_count = []
#     correct = []
#     response_time = []
#     skill_ids = []
#
#     q_cols = ["problem_id", "attempt_count", "correct", "ms_first_response", "skill_id"]
#     cols = [item_ids,attempt_count, correct, response_time, skill_ids]
#     print("\nAdding Metrics")
#     for i, row in tqdm(items.iterrows(), total=len(items)):
#         item_id = i
#         item_interactions = interactions.loc[interactions.problem_id == item_id][q_cols]
#         attempts = item_interactions["attempt_count"].mean()
#         attempt_count.append(attempts)
#
#         c = item_interactions["correct"].mean()
#         correct.append(c)
#
#         response = item_interactions["ms_first_response"].mean()
#         attempt_count.append(response)
#
#         skill_id = item_interactions["skill_id"].max()
#         skill_ids.append(skill_id)
#
#     for i, col in enumerate(q_cols):
#         items[q_cols[i]] = cols[i]
#
#
#     items.fillna(0,inplace = True)
#     items.replace({np.nan: 0}, inplace = True)
#     normalised = normalise(items[vector_features])
#     items[vector_features] = normalised[vector_features]
#     items.fillna(0, inplace = True)
#     items.replace({np.nan: 0}, inplace = True)
#
#     items.set_index("problem_id", inplace = True)
#
#     return items

