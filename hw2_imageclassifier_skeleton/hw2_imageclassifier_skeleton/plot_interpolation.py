import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
from typing import cast

from main import evaluate, Accuracy
from models import CNN
from torchvision import datasets, transforms

def get_identifier(seed, shuffle, batch_size, cnn, kernel, stride, optim_alg, lr, wd, rot_flip):
    identifier = hashlib.md5(
        str(
            (
                seed,
                shuffle,
                batch_size,
                cnn,
                kernel,
                stride,
                optim_alg,
                lr,
                wd,
                rot_flip,
            ),
        ).encode(),
    ).hexdigest()
    return identifier

def run_flatness_analysis(is_shuffled=False):
    seed = 47
    batch_size = 100
    kernel = 5
    stride = 1
    lr = 1e-2 if is_shuffled else 1e-3
    wd = 0.0
    cnn = True
    rot_flip = False
    optim_alg = "default"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    thrng = torch.Generator("cpu")
    thrng.manual_seed(seed)
    
    model = CNN(
        size=28,
        channels=[1, 100, 32],
        shapes=[300, 100, 10],
        kernel_size_conv=kernel,
        stride_size_conv=stride,
        kernel_size_pool=2,
        stride_size_pool=2
    )
    
    model.initialize(thrng)
    model = model.to(device)
    theta_0 = {k: v.clone() for k, v in model.state_dict().items()}

    ident = get_identifier(seed, is_shuffled, batch_size, cnn, kernel, stride, optim_alg, lr, wd, rot_flip)
    model_path = os.path.join("ptnnp", f"{ident}.ptnnp")
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}.")
        print(f"Ensure you ran: python main.py --cnn --num-epochs 100 {'--shuffle-label --lr 1e-2' if is_shuffled else ''}")
        return None, None, None

    theta_final = torch.load(model_path, map_location=device)

    dataset_train = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transforms.ToTensor())
    dataset_test = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transforms.ToTensor())
    
    if is_shuffled:
        thrng_data = torch.Generator("cpu")
        thrng_data.manual_seed(seed)
        shuffle_idx_train = torch.randperm(len(dataset_train), generator=thrng_data)
        dataset_train.targets = dataset_train.targets[shuffle_idx_train]
        
        thrng_data.manual_seed(seed)
        shuffle_idx_test = torch.randperm(len(dataset_test), generator=thrng_data)
        dataset_test.targets = dataset_test.targets[shuffle_idx_test]

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    alphas = np.linspace(0, 2, 21)
    train_losses = []
    test_losses = []
    
    for alpha in alphas:
        interpolated_sd = {}
        for k in theta_0.keys():
            interpolated_sd[k] = (1 - alpha) * theta_0[k] + alpha * theta_final[k]
        
        model.load_state_dict(interpolated_sd)
        train_loss = evaluate(model, criterion, loader_train, device=device)
        test_loss = evaluate(model, criterion, loader_test, device=device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
    return alphas, train_losses, test_losses

if __name__ == "__main__":
    alphas, train_orig, test_orig = run_flatness_analysis(is_shuffled=False)
    _, train_shuff, test_shuff = run_flatness_analysis(is_shuffled=True)

    if alphas is not None:
        plt.figure()
        
        plt.subplot(1, 2, 1)
        plt.plot(alphas, train_orig, label='Train Loss', marker='o', color='blue')
        plt.plot(alphas, test_orig, label='Test Loss (Gen)', marker='x', color='lightblue', linestyle='--')
        plt.axvline(x=0, color='gray', linestyle='--', label='theta_0')
        plt.axvline(x=1, color='black', linestyle='--', label='theta_final')
        plt.title('Original MNIST Flatness')
        plt.xlabel('Interpolation Coefficient')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(alphas, train_shuff, label='Train Loss', marker='s', color='red')
        plt.plot(alphas, test_shuff, label='Test Loss (Gen)', marker='x', color='salmon', linestyle='--')
        plt.axvline(x=0, color='gray', linestyle='--', label='theta_0')
        plt.axvline(x=1, color='black', linestyle='--', label='theta_final')
        plt.title('Shuffled Labels Flatness')
        plt.xlabel('Interpolation Coefficient')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('flatness_plot.png')
        plt.show()