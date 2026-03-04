#
import argparse
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import hashlib
import os
import numpy as np
import torch
import pandas as pd # type: ignore
import copy
from typing import Tuple, Dict, Union, List


def load(
    *,
    seed: int, shuffle: bool, batch_size: int, cnn: bool,
    kernel: int, stride: int, amprec: bool, optim_alg: str, lr: float,
    wd: float, ce: bool, rot_flip: bool,
) -> Tuple[float, pd.DataFrame]:
    R"""
    Load log.
    """
    #
    identifier = (
        hashlib.md5(
            str(
                (
                    seed, shuffle, batch_size, cnn, kernel, stride,
                    amprec, optim_alg, lr, wd, rot_flip,
                ),
            ).encode(),
        ).hexdigest()
    )

    #
    stderr = os.path.join("sbatch", "{:s}.stderr.txt".format(identifier))
    ptlog = os.path.join("ptlog", "{:s}.ptlog".format(identifier))

    #
    with open(stderr, "r") as file:
        #
        for line in file:
            #
            (key, val) = line.strip().split(": ")
            if key == "Elapsed":
                #
                (val, unit) = val.split(" ")
                if unit == "sec":
                    #
                    runtime = float(val)
                else:
                    # UNEXPECT:
                    # Unknown runtime unit.
                    raise RuntimeError("Unknown runtime unit.")

    #
    data_dict: Dict[str, Union[List[float], List[int], List[str]]]

    #
    data_dict = {}
    accs = np.array(torch.load(ptlog))
    n = len(accs)
    data_dict["Epoch"] = list(range(n))
    data_dict["Seed"] = [str(seed)] * n
    data_dict["Shuffle"] = ["Shuffle" if shuffle else "No-Shuffle"] * n
    data_dict["Batch Size"] = [str(batch_size)] * n
    data_dict["Model"] = (
        ["CNN" if cnn else "MLP"] * n
    )
    data_dict["Convolve"] = [", ".join([str(kernel), str(stride)])] * n
    data_dict["AMP"] = ["AMP" if amprec else "No-AMP"] * n
    data_dict["Optim"] = [optim_alg] * n
    data_dict["LR"] = [str(lr)] * n
    data_dict["WD"] = [str(wd)] * n

    #
    if ce:
        data_dict_train_ce = copy.deepcopy(data_dict)
        data_dict_train_ce["Cross Entropy"] = accs[:, 0].tolist()
        data_dict_train_ce["Case"] = ["Train CE"] * n
        return (runtime, pd.DataFrame(data_dict_train_ce))
    else:
        #
        data_dict_train_acc = copy.deepcopy(data_dict)
        data_dict_train_acc["Accuracy"] = accs[:, 1].tolist()
        data_dict_train_acc["Case"] = ["Train Acc"] * n
        data_dict_test_acc = copy.deepcopy(data_dict)
        data_dict_test_acc["Accuracy"] = accs[:, 2].tolist()
        data_dict_test_acc["Case"] = ["Test Acc"] * n
        return (
            (
                runtime,
                pd.concat(
                    (
                        pd.DataFrame(data_dict_train_acc),
                        pd.DataFrame(data_dict_test_acc),
                    ),
                    ignore_index=True,
                ),
            )
        )


def task_batch_size(ce: bool, /) -> None:
    R"""
    Batch size task.
    """
    #
    buf_runtime = []
    buf_frame = []
    for batch_size in (100, 500, 3000, 5000):
        #
        for lr in (1e-3, 1e-4, 1e-5):
            #
            (runtime, frame) = (
                load(
                    seed=47, shuffle=False, batch_size=batch_size, cnn=False,
                    kernel=5, stride=1, amprec=False,
                    optim_alg="sgd", lr=lr, wd=0.0, ce=ce, rot_flip=False,
                )
            )
            buf_runtime.append(runtime)
            buf_frame.append(frame)
    frame = pd.concat(buf_frame, ignore_index=True)

    #
    grids = (
        sns.relplot(
            data=frame, x="Epoch", y="Cross Entropy" if ce else "Accuracy",
            hue="Case", row="Batch Size", col="LR", style="Case", kind="line",
        )
    )
    figure = grids.figure
    figure.savefig(
        os.path.join(
            "figure", "batch_size_{:s}.png".format("ce" if ce else "acc"),
        ),
    )
    plt.close(figure)


def task_optimizer(ce: bool, /) -> None:
    R"""
    Optimizer task.
    """
    #
    buf_runtime = []
    buf_frame = []
    for optim_alg in ("sgd", "momentum", "nesterov", "adam"):
        #
        (runtime, frame) = (
            load(
                seed=47, shuffle=False, batch_size=100, cnn=False,                kernel=5, stride=1, amprec=False, optim_alg=optim_alg, lr=1e-3,
                wd=0.0, ce=ce, rot_flip=False,
            )
        )
        buf_runtime.append(runtime)
        buf_frame.append(frame)
    frame = pd.concat(buf_frame, ignore_index=True)

    #
    grids = (
        sns.relplot(
            data=frame, x="Epoch", y="Cross Entropy" if ce else "Accuracy",
            hue="Case", col="Optim", style="Case", kind="line",
        )
    )
    figure = grids.figure
    figure.savefig(
        os.path.join(
            "figure", "optimizer_{:s}.png".format("ce" if ce else "acc"),
        ),
    )
    plt.close(figure)


def task_regularization(ce: bool, /) -> None:
    R"""
    Regularization task.
    """
    #
    buf_runtime = []
    buf_frame = []
    for l2lambda in (1.0, 0.1, 0.01):
        #
        (runtime, frame) = (
            load(
                seed=47, shuffle=False, batch_size=100, cnn=False,                kernel=5, stride=1, amprec=False, optim_alg="sgd", lr=1e-3,
                wd=l2lambda, ce=ce, rot_flip=False,
            )
        )
        buf_runtime.append(runtime)
        buf_frame.append(frame)
    frame = pd.concat(buf_frame, ignore_index=True)

    #
    grids = (
        sns.relplot(
            data=frame, x="Epoch", y="Cross Entropy" if ce else "Accuracy",
            hue="Case", col="WD", style="Case", kind="line",
        )
    )
    figure = grids.figure
    figure.savefig(
        os.path.join(
            "figure", "regularization_{:s}.png".format("ce" if ce else "acc"),
        ),
    )
    plt.close(figure)


def task_cnn(ce: bool, /) -> None:
    R"""
    CNN task.
    """
    #
    buf_runtime = []
    buf_frame = []
    for (kernel, stride) in ((5, 1), (3, 3), (14, 1)):
        #
        (runtime, frame) = (
            load(
                seed=47, shuffle=False, batch_size=100, cnn=True,                kernel=kernel, stride=stride, amprec=False,
                optim_alg="default", lr=1e-3, wd=0.0, ce=ce, rot_flip=False,
            )
        )
        buf_runtime.append(runtime)
        buf_frame.append(frame)
    frame = pd.concat(buf_frame, ignore_index=True)

    #
    grids = (
        sns.relplot(
            data=frame, x="Epoch", y="Cross Entropy" if ce else "Accuracy",
            hue="Case", col="Convolve", style="Case", kind="line",
        )
    )
    figure = grids.figure
    figure.savefig(
        os.path.join(
            "figure", "cnn_{:s}.png".format("ce" if ce else "acc"),
        ),
    )
    plt.close(figure)


def task_shuffle_cnn(ce: bool, /) -> None:
    R"""
    CNN task.
    """
    #
    buf_runtime = []
    buf_frame = []

    (runtime, frame) = (
        load(
            seed=47, shuffle=True, batch_size=100, cnn=True,            kernel=5, stride=1, amprec=False,
            optim_alg="default", lr=1e-2, wd=0.0, ce=ce, rot_flip=False,
        )
    )
    buf_runtime.append(runtime)
    buf_frame.append(frame)
    frame = pd.concat(buf_frame, ignore_index=True)

    #
    grids = (
        sns.relplot(
            data=frame, x="Epoch", y="Cross Entropy" if ce else "Accuracy",
            hue="Case", col="Convolve", style="Case", kind="line",
        )
    )
    figure = grids.figure
    figure.savefig(
        os.path.join(
            "figure", "shuffle_cnn_{:s}.png".format("ce" if ce else "acc"),
        ),
    )
    plt.close(figure)


def main(*ARGS):
    R"""
    Main.
    """
    # YOU SHOULD FILL IN THIS FUNCTION
    ...

    #
    parser = argparse.ArgumentParser(description="Visualization Execution")
    parser.add_argument(
        "--ce",
        action="store_true", help="Visualize training cross entropy loss.",
    )
    parser.add_argument(
        "--minibatch",
        action="store_true", help="Visualize minibatch task.",
    )
    parser.add_argument(
        "--optimizer",
        action="store_true", help="Visualize optimizer task.",
    )
    parser.add_argument(
        "--regularization",
        action="store_true", help="Visualize regularization task.",
    )
    parser.add_argument(
        "--cnn",
        action="store_true", help="Visualize CNN task.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle training label data.",
    )
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    # Parse the command line arguments.
    ce = args.ce
    minibatch = args.minibatch
    optimizer = args.optimizer
    regularization = args.regularization
    cnn = args.cnn
    shuffle = args.shuffle

    #
    if not os.path.isdir("figure"):
        #
        os.makedirs("figure")

    #
    if minibatch:
        #
        task_batch_size(ce)

    #
    if optimizer:
        #
        task_optimizer(ce)

    #
    if regularization:
        #
        task_regularization(ce)

    #
    if cnn:
        #
        task_cnn(ce)

    #
    if shuffle:
        #
        task_shuffle_cnn(ce)


#
if __name__ == "__main__":
    #
    main()