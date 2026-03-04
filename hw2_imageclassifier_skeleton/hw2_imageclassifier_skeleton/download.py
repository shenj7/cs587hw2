#
import argparse
import os
import torchvision # type: ignore


def main(*ARGS) -> None:
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Download Execution")
    parser.add_argument(
        "--source",
        type=str, required=False, default=os.path.join("data", "mnist"),
        help="Source directory for data.",
    )
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    #
    source = args.source

    #
    torchvision.datasets.MNIST(source, train=True, download=True)


#
if __name__ == "__main__":
    #
    main()