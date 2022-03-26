from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from Objects.compute_embeddings import CalcEmbeddings


def write_data_to_board(xs, run_name, metadata=[], headers=[]):
    writer = SummaryWriter(log_dir="runs/" + run_name)
    if metadata == [] or headers == "":
        writer.add_embedding(xs)
    else:
        writer.add_embedding(xs, metadata=metadata, metadata_header = headers)
    writer.flush()

def add_embeddings_to_tensorboard(data_file,model_file, file_name):
    train_embedder = CalcEmbeddings(data_file, model_file)
    embeddings, metadata = train_embedder.get_embeddings()

    print("# of Embeddings : ", len(embeddings))
    run_name = datetime.now().strftime("%b%d_%H-%M-%S") + "_" + file_name

    write_data_to_board(embeddings,run_name, metadata=metadata, headers = ["Q_ID","Skill ID","Skill Name"])


if __name__ == '__main__':
    add_embeddings_to_tensorboard("../skill_builder_data.csv", "questions_dataset_Mar10_16-48-43.pth", "training_data")
    add_embeddings_to_tensorboard("../non_skill_builder_data_new.csv", "questions_dataset_Mar10_16-48-43.pth", "test_data")
