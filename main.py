import os
import sys
import csv
import time
import numpy as np
from dataset import MIL_Survival_Dataset
from options import parse_args
from util import get_split_loader, set_seed
from loss import define_loss
from optimizer import define_optimizer
from scheduler import define_scheduler
from datetime import datetime

# ==

# def get_commit_hash(repo_path):
#     try:
#         head_file_path = os.path.join(repo_path, '.git', 'HEAD')
#         with open(head_file_path, 'r') as file:
#             ref = file.read().strip()

#         if ref.startswith('ref: '):
#             ref_path = os.path.join(repo_path, '.git', ref[5:])
#             with open(ref_path, 'r') as file:
#                 commit_hash = file.read().strip()
#             return commit_hash
#         return ref
#     except Exception as error:
#         print(f"Exception: {error}")


class FlushFile:
    def __init__(self, file):
        self.file = file

    def write(self, data):
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()


def current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main(args):
    start_time = datetime.now()
    set_seed(args.seed)

    results_dir = f"./results/{args.dataset}/model-[{args.model}]-[{args.fusion}]-[{args.alpha}]-[{time.strftime('%Y-%m-%d]-[%H-%M-%S')}]"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    log_file_path = os.path.join(results_dir, '___logging.txt')
    log_file_handle = open(log_file_path, 'w')
    sys.stdout = FlushFile(log_file_handle)

    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_epoch_list = ["best epoch"]
    best_score_list = ["best cindex"]
    repo_path = os.getcwd()
    commit_hash = get_commit_hash(repo_path)

    # print("=======================================")
    print("Parameters:", vars(args))
    # print("Git info:", commit_hash)
    # print("=======================================")

    temp_time = current_time()

    for fold in range(5):
        dataset = MIL_Survival_Dataset(
            csv_path=f"./csv/{args.dataset}_all_clean.csv",
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.data_root_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )
        split_dir = os.path.join("./splits", args.which_splits, args.dataset)
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path=f"{split_dir}/splits_{fold}.csv"
        )
        train_loader = get_split_loader(
            train_dataset,
            training=True,
            weighted=args.weighted_sample,
            modal=args.modal,
            batch_size=args.batch_size,
        )
        val_loader = get_split_loader(
            val_dataset, modal=args.modal, batch_size=args.batch_size
        )
        print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

        if args.model == "AdaMHF":
            from AdaMHF import AdaMHF
            from engine import Engine

            print(train_dataset.omic_sizes)
            model_config = {
                "omic_sizes": train_dataset.omic_sizes,
                "n_classes": 4,
                "fusion": args.fusion,
                "model_size": args.model_size,
                "alpha": args.F_alpha,
                "beta": args.F_beta,
                "tokenS": args.tokenS,
                "GT": args.GT,
                "PT": args.PT,
                "Rate": args.Rate,
                "pos": args.pos,
            }
            model = AdaMHF(**model_config)
            criterion = define_loss(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            engine = Engine(args, results_dir, fold)

        score, epoch = engine.learning(
            temp_time, model, train_loader, val_loader, criterion, optimizer, scheduler, args.dataset
        )
        best_epoch_list.append(epoch)
        best_score_list.append(score)

    best_epoch_list.extend(["~", "~"])
    best_score_list.append(np.mean(best_score_list[1:6]))
    best_score_list.append(np.std(best_score_list[1:6]))

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    csv_file_path = os.path.join(results_dir, "results.csv")
    elapsed_time_list = [elapsed_time] * 8

    with open(csv_file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(best_epoch_list)
        writer.writerow(best_score_list)
        writer.writerow(elapsed_time_list)

    mean_score = np.mean(best_score_list[1:6])
    new_dir_name = f"{results_dir}_{mean_score:.2f}__{args.modality}__[{args.GT}_{args.PT}]__[{args.lr}]_{args.weight_decay}]"
    os.rename(results_dir, new_dir_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)

