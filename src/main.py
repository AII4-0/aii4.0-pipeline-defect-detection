import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="DVC defect detection",
    )
    # Add subcommand
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    # Add prepare subcommand
    subparsers.add_parser(
        "prepare",
        help="Prepare the dataset",
    )

    # Add train subcommand
    subparsers.add_parser(
        "train",
        help="Train the model",
    )

    # Add evaluate subcommand
    subparsers.add_parser(
        "evaluate",
        help="Evaluate the model",
    )

    # Add export subcommand
    subparsers.add_parser(
        "export",
        help="Export the model to ONNX format",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run subcommand
    if args.subcommand == "prepare":
        from src.prepare import prepare

        prepare()
    elif args.subcommand == "train":
        from src.train import train

        train()
    elif args.subcommand == "evaluate":
        from src.evaluate import evaluate

        evaluate()
    elif args.subcommand == "export":
        from src.export import export

        export()
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
