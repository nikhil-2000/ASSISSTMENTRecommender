from tqdm import trange

from Train_Test.random_walk_model import *
from Metrics.Random_Model import RandomChoiceMetrics
from Metrics.Random_Walk import RandomWalkMetrics
from Metrics.Same_Skill import SameSkillMetrics
from Metrics.Collab_Filtering import CollabFilteringMetrics

from prettytable import PrettyTable

def get_random_user(memory, interactions, users):
    total_questions = 0
    j = 0
    while total_questions < 3:
        user_id = pd.Series(users.index.values).sample(1).item()
        if not user_id in memory:
            condition = interactions["user_id"] == user_id
            memory[user_id] = interactions.loc[condition]

        user_questions = memory[user_id]

        total_questions = len(user_questions)
        # j += 1
        # if j > 3:
        #     print(j)

    user_questions = user_questions.reset_index()
    return user_questions, total_questions, memory

def run_metrics(filename, model, samples = 1000, tests = 1000, size = 0):
    random_walk_metrics = RandomWalkMetrics(filename, model, size = size)
    metadata = random_walk_metrics.metadata
    dataset = random_walk_metrics.embedder.dataset
    users, interactions = dataset.users_df, dataset.interactions_df
    same_skill_metrics = SameSkillMetrics(metadata)
    random_metrics = RandomChoiceMetrics(metadata)
    collab_filtering_metrics = CollabFilteringMetrics(filename, size= size)
    memory = {}

    models = [random_walk_metrics, same_skill_metrics, random_metrics, collab_filtering_metrics]

    search_size = math.floor(len(metadata) * 10 ** -2)

    same_count = 0

    model_names = ["Random Walk", "Same Skill", "Random Choice", "Collab Filtering"]

    for i in trange(tests):
        # Pick Random User
        user_questions, total_questions ,memory = get_random_user(memory, interactions, users)
        # Generate Anchor Positive
        a_idx, p_idx = random.sample(range(0, total_questions), 2)
        anchor = user_questions.loc[a_idx]
        anchor_id = anchor.problem_id.item()
        anchor_skill = anchor.skill_id.item()

        positive = user_questions.loc[p_idx]
        positive_id = positive.problem_id.item()
        positive_skill = positive.skill_id.item()

        without_positives = metadata[~metadata.problem_id.isin(user_questions.problem_id.unique())]
        random_ids = without_positives.problem_id.sample(samples).to_list()
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


    output = PrettyTable()
    output.field_names = ["Model", "Hitrate", "Mean Rank"]
    for i, name in enumerate(model_names):
        m = models[i]

        hr = m.hitrate(tests)
        mr = m.mean_rank()
        output.add_row([name, hr, mr])


    print(output)
    print("Same skill ID:", same_count)



if __name__ == '__main__':

    run_metrics("../skill_builder_data.csv", "../updated_objects_Mar23_23-39-20.pth", samples = 1000, tests = 1000, size = 0000)
    run_metrics("../non_skill_builder_data_new.csv", "../updated_objects_Mar23_23-39-20.pth", samples = 1000, tests = 1000, size = 0000)



