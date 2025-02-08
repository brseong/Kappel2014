from ast import arg
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
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=50)
parser.add_argument("--population", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--feature_map", type=int, nargs="+", default=[0, 3])
parser.add_argument("--learning_rate", type=float, default=1e-1)
parser.add_argument("--tau", type=int, default=10)
parser.add_argument("--refractory_period", type=int, default=5)
parser.add_argument("--num_paths", type=int, default=4)
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
refractory_period = args.refractory_period
num_paths = args.num_paths
in_features = 28 * 28 * populations
out_features = len(feature_map) * refractory_period  # 10 classes default
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

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
    },
)

SpikeLoader = DataLoader[Dataset[UInt8[th.Tensor, "Timesteps Population 28 28"]]]

if __name__ == "__main__":
    th.no_grad()
    encode_data = partial(encode_data, num_steps=num_steps, population=populations)
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
        refractory_period,
        num_paths,
    ).to(device)
    wandb.watch(net)  # type: ignore

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
                wandb.log(
                    {
                        f"p(x|s_{k}=1)": wandb.Image(
                            decode_population(
                                net.log_likelihood_afferent[:, k]
                                .exp()
                                .view(populations, 28, 28)
                            )
                        )
                        for k in range(out_features)
                    }
                    | {
                        f"p(s_{k}=1)": net.log_prior[k].exp().item()
                        for k in range(out_features)
                    }
                    | {
                        "p(s_{t}|s_{t-1}=1)": wandb.Image(
                            net.log_likelihood_lateral.exp()
                        )
                    }
                )
