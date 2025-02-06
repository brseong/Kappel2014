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

num_steps = 50
populations = 2
num_epochs = 1
batch_size = 32
num_workers = 4
feature_map = [0, 3]
# feature_map = [0, 3, 4]
# feature_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
out_features = len(feature_map)  # 10 classes default
learning_rate = 1e0
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

wandb.init(
    project="kappel2014",
    config={
        "num_steps": num_steps,
        "population": populations,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "out_features": out_features,
        "learning_rate": learning_rate,
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

    net = HMM(28 * 28 * populations, out_features, learning_rate, populations).to(
        device
    )
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
            if i % 10 == 0:
                for k in range(out_features):
                    log_likelihood_k_afferent = net.log_likelihood_afferent[:, k]

                    wandb.log(
                        {
                            f"p(x|s_{k}=1)": wandb.Image(
                                decode_population(
                                    log_likelihood_k_afferent.exp().view(
                                        populations, 28, 28
                                    )
                                )
                            ),
                        }
                    )
                wandb.log(
                    {
                        "p(s_{t}|s_{t-1}=1)": wandb.Image(
                            net.log_likelihood_lateral.exp()
                        ),
                        f"p(s)": wandb.Image(
                            net.log_prior.exp().unsqueeze(-1),
                        ),
                    }
                )
