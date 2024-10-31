import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")

    arg_parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/data/",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    arg_parser.add_argument(
        "--which_splits",
        type=str,
        default="5foldcv",
    )

    arg_parser.add_argument(
        "--dataset",
        type=str,
        default="tcga_xxxx",
    )


    arg_parser.add_argument(
        "--model",
        type=str,
        default="AdaMHF",
        help="Model type",
    )


    arg_parser.add_argument("--alpha", type=float, default=1)

    arg_parser.add_argument(
        "--optimizer",
        type=str,
        choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"],
        default="Adam"
    )

    arg_parser.add_argument(
        "--scheduler",
        type=str,
        default="None"
    )

    arg_parser.add_argument("--num_epoch", type=int, default=30)

    arg_parser.add_argument("--batch_size", type=int, default=1)

    arg_parser.add_argument(
        "--loss",
        type=str,
        default="nll_surv",
        help="Slide-level classification loss function (default: ce)",
    )


    arg_parser.add_argument("--GT", type=float, default=0.5, help="Temperature of gene token selection (current num*T)")

    arg_parser.add_argument("--PT", type=float, default=0.5,
                            help="Temperature of image token selection (current num*T)")

    arg_parser.add_argument("--lr", type=float, default=2e-4)

    arg_parser.add_argument("--weight_decay", type=float, default=1e-5)

    arg_parser.add_argument("--tokenS", type=str, choices=["both", "G", "P", "N"], default="both")

    arg_parser.add_argument(
        "--fusion",
        type=str,
        default="LMF",
        help="Modality fusion strategy",
    )

    arg_parser.add_argument("--Rate", type=float, default=1e-2, help="Fusion rate")

    arg_parser.add_argument("--modality", type=str, choices=["Both", "G", "P"], default="Both", help="Missing modality")

    arg_parser.add_argument("--pos", type=str, default="epeg", help="Positional encoder")

    return arg_parser.parse_args()
