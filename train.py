from functools import partial
import pdb
import torch as th
from utils.Nessler2009 import Nessler2009
from utils.coding import encode_data, decode_population
from utils.module import HMM
from jaxtyping import UInt8
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--population", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--feature_map", type=int, nargs="+", default=[0, 3, 4])
parser.add_argument("--learning_rate", type=float, default=5e-1)
parser.add_argument("--tau", type=int, default=2)
parser.add_argument("--stdp_window", type=int, default=2)
parser.add_argument("--refractory_period", type=int, default=1)
parser.add_argument("--num_paths", type=int, default=4)
parser.add_argument("--target_rate", type=int, default=10)
args = parser.parse_args()


num_steps = args.num_steps
populations = args.population
num_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.num_workers
feature_map = args.feature_map
# feature_map = [0, 3, 4]
# feature_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
learning_rate = args.learning_rate
tau = args.tau
stdp_window = args.stdp_window
refractory_period = args.refractory_period
num_paths = args.num_paths
target_rate = args.target_rate
in_features = 28 * 28 * populations
out_features = 1 + len(feature_map) * 10  # 10 classes default. one for start state.
device = th.device("cuda:3" if th.cuda.is_available() else "cpu")

wandb.init(
    project="kappel2014",
    config={
        "num_steps": num_steps,
        "population": populations,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "in_features": in_features,
        "out_features": out_features,
        "learning_rate": learning_rate,
        "tau": tau,
        "refractory_period": refractory_period,
        "num_paths": num_paths,
        "stdp_window": stdp_window,
        "feature_map": feature_map,
        "target_rate": target_rate,
    },
)

SpikeLoader = DataLoader[Dataset[UInt8[th.Tensor, "Timesteps Population 28 28"]]]

if __name__ == "__main__":
    th.no_grad()
    encode_data = partial(
        encode_data,
        num_steps=num_steps,
        population=populations,
        prob=target_rate / num_steps,
    )
    data_train = MNIST(".", download=True, train=True, transform=encode_data)
    data_test = MNIST(".", download=True, train=False, transform=encode_data)
    train_loader = SpikeLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = SpikeLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    net = HMM(
        in_features,
        out_features,
        learning_rate,
        populations,
        tau,
        stdp_window,
        refractory_period,
        num_paths,
        batch_size,
    ).to(device)
    wandb.watch(net, log="parameters")  # type: ignore

    ewma = 0.5  # To inspect the clustering of the likelihoods
    for epoch in range(num_epochs):
        data: UInt8[th.Tensor, "Batch Timesteps Population 28 28"]
        for i, (data, target) in tqdm(enumerate(iter(train_loader))):
            mask = th.zeros_like(target, dtype=th.bool)
            for v in feature_map:
                mask |= target == v
            data = data[mask, :]
            target = target[mask]
            if data.shape[0] == 0:
                continue
            data = data.view(data.shape[0], num_steps, -1).to(device)
            target = target.to(device)
            pred = net(data)
            if i % 1 == 0:
                log_data = {}
                # Create heatmaps for p(x|s_k=1)
                for k in range(out_features):
                    arr = (
                        net.log_likelihood_afferent[:, k]
                        .exp()
                        .view(populations, 28, 28)
                        .cpu()
                    )
                    decoded_arr = decode_population(arr)
                    fig, ax = plt.subplots()
                    im = ax.imshow(decoded_arr, cmap="hot")
                    plt.colorbar(im, ax=ax)
                    log_data[f"$p(x|s_{k}=1)$"] = wandb.Image(fig)
                    plt.close(fig)

                # Log p(s_k=1) as scalar values
                for k in range(out_features):
                    log_data[f"$p(s_{k}=1)$"] = net.log_prior[k].item()

                # Create heatmap for p(s_{t}|s_{t-1}=1)
                fig, ax = plt.subplots()
                lateral = net.log_likelihood_lateral.exp().cpu().numpy()
                im = ax.imshow(lateral, cmap="hot")
                plt.colorbar(im, ax=ax)
                log_data["$p(s_{t}|s_{t-1}=1)$"] = wandb.Image(fig)
                plt.close(fig)

                # Create heatmap for generated output
                fig, ax = plt.subplots()
                gen_data = (
                    net(batch_size=1)
                    .float()
                    .mean(dim=1)
                    .view(populations, 28, 28)
                    .cpu()
                )
                decoded_gen = decode_population(gen_data)
                im = ax.imshow(decoded_gen, cmap="hot")
                plt.colorbar(im, ax=ax)
                log_data["generated"] = wandb.Image(fig)
                plt.close(fig)

                wandb.log(log_data)
