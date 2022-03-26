import math
import random
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm, trange
from Metrics.Collab_Filtering import CollabFilteringMetrics

"""
Collaborative filtering attempt
Will split skills in categories
For each user I will gather how they perform on each category id
Then find the nearest user based on average score and attempt times
Pick one of the nearby users, then choose some of the assignments they have completed
"""


class User:

    def __init__(self, subset_df: pd.DataFrame, all_skills, train_interactions: pd.DataFrame,
                 test_interactions: pd.DataFrame):
        self.id = subset_df.iloc[0]["user_id"]
        d = {k: 0 for k in all_skills}
        subset_df = subset_df.sort_values(by=["skill_id"])
        for i, row in subset_df.iterrows():
            skill, score = row["skill_id"], row["correct"]
            d[skill] = score

        self.scores = np.array(list(d.values()))
        self.interactions = train_interactions
        self.test = test_interactions

    def get_vector(self):
        # print(self.scores.transpose().shape)
        return self.scores.transpose()

    def has_skills(self):
        return any(self.scores)

    def get_sample(self):
        q = self.test.sample(1)
        # other_qs = self.test[self.test.assignment_id == q.assignment_id.item()]
        skill_id = q.skill_id.item()

        return q.problem_id.item(), skill_id

    def get_questions_by_skill_train(self, skill_id, n):
        qs = self.interactions[self.interactions.skill_id == skill_id]
        remaining = n - len(qs)
        if len(self.interactions) >= remaining > 0:
            random_qs = self.interactions.sample(remaining)
            return qs.problem_id.to_list() + random_qs.problem_id.to_list()
        elif len(self.interactions) < remaining:
            random_qs = self.interactions
            zeroes = [0] * (n - len(random_qs) - len(qs))
            return qs.problem_id.to_list() + random_qs.problem_id.to_list() + zeroes
        else:
            return qs.sample(n).problem_id.to_list()

    def get_questions_by_skill_test(self, skill_id, n):
        qs = self.test[self.test.skill_id == skill_id]
        remaining = n - len(qs)
        if len(self.test) >= remaining > 0:
            random_qs = self.test.sample(remaining)
            return qs.problem_id.to_list() + random_qs.problem_id.to_list()
        elif len(self.test) < remaining:
            random_qs = self.test
            zeroes = [0] * (n - len(random_qs) - len(qs))
            return qs.problem_id.to_list() + random_qs.problem_id.to_list() + zeroes
        else:
            return qs.sample(n).problem_id.to_list()

    def rank_questions(self, ids, anchor_data):
        ids = list(ids)
        anchor_id, anchor_skill = anchor_data
        same_skill_qs = set(self.interactions.loc[self.interactions["skill_id"] == anchor_skill].problem_id)
        in_samples = set(ids).intersection(same_skill_qs)
        ordering = []
        for id in in_samples:
            ordering.append(id)
            ids.remove(id)

        return ordering + ids


def getSkillID(ids):
    if isinstance(ids, int) or isinstance(ids, float):
        return ids

    ids = ids.split(",")
    return ids[0]

def read_data(filename):
    dataset = pd.read_csv(filename, encoding="latin-1")

    dataset['skill_id'] = dataset['skill_id'].fillna(0)
    dataset["skill_id"] = dataset["skill_id"].apply(getSkillID)
    dataset["skill_id"] = dataset["skill_id"].apply(int)
    dataset = dataset[dataset["skill_id"] > 0]

    user_performance = dataset.groupby(["user_id", "skill_id"])["correct"].mean()
    user_performance = pd.DataFrame(user_performance)
    user_performance.reset_index(inplace=True)

    return dataset, user_performance


def create_users(user_performance, dataset):
    all_users = user_performance.user_id.unique()
    all_skills = user_performance.skill_id.unique()

    users = []
    # print(len(all_skills))
    # skill_vectors = pd.DataFrame(columns=["user_id"] + list(all_skills))
    print("Creating Users")
    for user in tqdm(all_users):
        user_scores = user_performance[user_performance["user_id"] == user]
        interactions = dataset[dataset.user_id == user]
        if len(interactions) > 10:
            new_user = User(user_scores, all_skills, interactions, interactions)
            users.append(new_user)
    # user_performance.reset_index(inplace = True)

    users = list(filter(lambda x: x.has_skills(), users))

    return users, all_skills
# user_ids = list(map(lambda x: x.id, users))
# skill_vectors["user_id"] = user_ids
# skill_vectors = skill_vectors.set_index("user_id")
# for user in users:
#     skill_vectors.at[user.id] = user.scores
#
# mask = (skill_vectors != 0).any(axis=0)
# skill_vectors = skill_vectors.loc[:, mask]
def get_user_scores(users, all_skills):
    score_matrix = np.zeros((len(all_skills), len(users)))
    ids_to_idx = {}
    for idx, user in enumerate(users):
        ids_to_idx[user.id] = idx
        score_matrix[:, idx] = user.get_vector()

    return score_matrix, ids_to_idx

def hitrate(score_matrix_ids, dataset, users):
    score_matrix, ids_to_idx = score_matrix_ids
    dists = pairwise_distances(score_matrix.transpose())
    hits = 0
    samples = 5000
    search_size = 30 #math.floor(len(dataset.problem_id.unique()) * 10**-3)

    print("Running Tests")
    for i in trange(samples):
        user = random.choice(users)
        anchor, skill_id = user.get_sample()
        positive, _ = user.get_sample()


        idx = ids_to_idx[user.id]
        dists_from_users = dists[:, idx]
        sorted_indexes = np.argsort(dists_from_users)
        best_idx = sorted_indexes[1]
        similar_user = users[best_idx]

        predicted_questions_train = similar_user.get_questions_by_skill_train(skill_id, search_size)

        if anchor in predicted_questions_train: predicted_questions_train.remove(anchor)


        if positive in predicted_questions_train:
            hits += 1



    return 100 * hits / samples, search_size


def MRR(score_matrix_ids, dataset, users, search_size = 1000, tests = 1000):
    score_matrix, ids_to_idx = score_matrix_ids
    dists = pairwise_distances(score_matrix.transpose())


    ranks = {"baseline": [], "random": []}

    for i in trange(tests):
        total_questions = 0
        j = 0
        while total_questions < 10:
            user = random.choice(users)

            user_questions = user.interactions

            total_questions = len(user_questions)
            # j += 1
            # if j > 3:
            #     print(j)

        user_questions = user_questions.reset_index()
        a_idx, p_idx = random.sample(range(0, total_questions), 2)
        anchor = user_questions.loc[a_idx]
        anchor_id = anchor.problem_id.item()
        anchor_skill = anchor.skill_id.item()

        positive = user_questions.loc[p_idx]
        positive_id = positive.problem_id.item()
        random_ids = dataset.problem_id.sample(search_size).to_list()
        all_ids = [positive_id] + random_ids
        random.shuffle(all_ids)

        idx = ids_to_idx[user.id]
        dists_from_users = dists[:, idx]
        sorted_indexes = np.argsort(dists_from_users)
        best_idx = sorted_indexes[1]
        similar_user = users[best_idx]

        similar_user_ranks = similar_user.rank_questions(all_ids, [anchor_id, anchor_skill])

        random_order = [positive_id] + random_ids
        random.shuffle(random_order)

        predictions = [similar_user_ranks, random_order]
        methods = list(ranks.keys())
        for idx, prediction in enumerate(predictions):

            if positive_id in prediction:
                rank = prediction.index(positive_id)
            else:
                rank = search_size + 1


            ranks[methods[idx]].append(rank)

        # times[i] = np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3])

    # print([int(t) for t in sum_of_times])
    ranks = {k: np.mean(v) for k, v in ranks.items()}
    return ranks

def get_random_user(interactions, users):
    total_questions = 0
    j = 0
    while total_questions < 3:
        user = random.choice(users)


        total_questions = len(user.interactions)
        # j += 1
        # if j > 3:
        #     print(j)

    user_questions = user.interactions.reset_index()
    return user_questions, total_questions


def run_metrics(filename, samples = 1000, tests = 1000, size = 0):
    collabFilteringMetrics = CollabFilteringMetrics(filename, size = size, tests=tests)
    interactions = collabFilteringMetrics.data
    users = collabFilteringMetrics.users

    models = [collabFilteringMetrics]

    search_size = 1000 #math.floor(len(collabFilteringMetrics) * 10 ** -2)

    same_count = 0

    model_names = ["Collab Filtering"]

    for i in trange(tests):
        # Pick Random User
        user_questions, total_questions = get_random_user(interactions, users)
        # Generate Anchor Positive
        a_idx, p_idx = random.sample(range(0, total_questions), 2)
        anchor = user_questions.loc[a_idx]
        anchor_id = anchor.problem_id.item()
        anchor_skill = anchor.skill_id.item()

        positive = user_questions.loc[p_idx]
        positive_id = positive.problem_id.item()
        positive_skill = positive.skill_id.item()

        without_anchor = interactions[interactions.problem_id != anchor_id]
        without_positive = without_anchor[without_anchor.problem_id != positive_id]
        random_ids = without_positive.problem_id.sample(samples).to_list()
        all_ids = random_ids + [positive_id]
        random.shuffle(all_ids)

        # Find n Closest
        for i,m in enumerate(models):
            top_n = m.top_n_questions(anchor, search_size)
            ranking = m.rank_questions(all_ids, anchor)

            set_prediction = set(top_n)
            if positive_id in set_prediction:
                m.hits += 1

            rank = ranking.index(positive_id)

            m.ranks.append(rank)

        if positive_skill == anchor_skill:
            same_count += 1
        # else:
        #     print(same_skill_metrics.ranks[-1])

    avgs = collabFilteringMetrics.times.mean(axis = 0)
    print("Times :", avgs)

    for i, name in enumerate(model_names):
        m = models[i]
        print("Model :", name)
        print("Hitrate :", m.hitrate(tests))
        print("Mean Rank :", m.mean_rank())
        print()


    print("Same skill ID:", same_count)



if __name__ == '__main__':
    """
    train, train_performance = read_data("skill_builder_data.csv")
    train_users, train_skills = create_users(train_performance, train)
    score_matrix_ids = get_user_scores(train_users, train_skills)
    train_hits, train_search_size = hitrate(score_matrix_ids, train, train_users)
    train_ranks = MRR(score_matrix_ids, train, train_users, search_size=1000, tests = 5000)


    test, test_performance = read_data("non_skill_builder_data_new.csv")
    test_users, test_skills = create_users(test_performance, test)
    score_matrix_ids = get_user_scores(test_users, test_skills)
    test_hits, test_search_size = hitrate(score_matrix_ids, test, test_users)
    test_ranks = MRR(score_matrix_ids, test, test_users, search_size=1000, tests = 5000)

    print("Training Dataset")
    print("Assignment Hitrate from Train:", train_hits)
    print("Dataset Search Size:", train_search_size)
    print("Mean Rank of Positives:", train_ranks)
    print()

    print("Test Dataset")
    print("Assignment Hitrate from Test:", test_hits)
    print("Dataset Search Size:", test_search_size)
    print("Mean Rank of Positives:", test_ranks)



    # train_q_ids = train.problem_id.unique()
    # test_q_ids = test.problem_id.unique()
    # print("Train Qs:", len(train_q_ids))
    # print("Test Qs:", len(test_q_ids))
    #
    # same_ids = set(train_q_ids).intersection(set(test_q_ids))
    # print("Intersection of Datasets: ", len(same_ids))

#   Try picking a random user rather than similar
#   Look into meanings behind metrics further

    """
    run_metrics("../skill_builder_data.csv", samples=1000, tests=50)