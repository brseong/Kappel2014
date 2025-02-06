import pdb
from typing import Literal
import torch as th
from torchvision.transforms import ToTensor  # type: ignore
from PIL.Image import Image
from jaxtyping import Float, UInt8, Int

to_tensor = ToTensor()


def poisson_spike_per_cls(
    y_classes: UInt8[th.Tensor, "1 28 28"],
    num_steps: int = 50,
    prob: float = 4e-2,
    population: int = 2,
) -> UInt8[th.Tensor, "Num_steps Populations 28 28"]:
    """Generate Poisson spikes for each class in the input tensor. Only one class is active for the series of spikes.

    Returns:
        _type_: _description_
    """
    y_indices = y_classes.unsqueeze_(0).repeat(
        num_steps, 1, 1, 1
    )  # (t_steps, 1, 28, 28)
    poisson = th.distributions.poisson.Poisson(rate=prob)
    spikes = th.zeros(
        (num_steps, population, *y_indices.shape[2:]), dtype=th.uint8
    )  # (t_steps, Population, 28, 28)
    _spikes = poisson.sample((num_steps, 1, *y_indices.shape[2:])).to(
        th.uint8
    )  # (t_steps, 1, 28, 28)
    return spikes.scatter_(dim=1, index=y_indices, src=_spikes)


def temporal_spike_per_cls(
    y_classes: UInt8[th.Tensor, "1 28 28"],
    num_steps: int = 50,
    population: int = 2,
) -> UInt8[th.Tensor, "Num_steps Populations 28 28"]:
    """Generate temporal spikes for each class in the input tensor. Only one class is active for the series of spikes.

    Returns:
        _type_: _description_
    """
    y_indices = y_classes.unsqueeze_(0).repeat(
        num_steps, 1, 1, 1
    )  # (t_steps, 1, 28, 28)
    spikes = th.zeros(
        (num_steps, population, *y_indices.shape[2:]), dtype=th.uint8
    )  # (t_steps, Population, 28, 28)
    raise NotImplementedError("Temporal spikes are not implemented yet.")
    return spikes


def encode_data(
    x: Image | th.Tensor,
    num_steps: int = 50,
    population: int = 2,
    method: Literal["poisson", "temporal"] = "poisson",
) -> UInt8[th.Tensor, "Num_steps Populations 28 28"]:
    """Encode data into Poisson spikes, where each pixel is represented by a population of neurons.

    Returns:
        _type_: _description_
    """
    if not isinstance(x, th.Tensor):
        x = to_tensor(x)
    x *= population  # (1, 28, 28)
    x = (
        x.floor_().clamp_(0, population - 1).to(dtype=th.int64)
    )  # Split the image into population parts: [0,...,population-1]

    if method == "poisson":
        return poisson_spike_per_cls(x, num_steps=num_steps, population=population)
    elif method == "temporal":
        return temporal_spike_per_cls(x, num_steps=num_steps, population=population)
    else:
        raise ValueError(f"Unknown method: {method}")


def decode_population(
    x: UInt8[th.Tensor, "Populations 28 28"],
) -> Float[th.Tensor, "28 28"]:
    """Integrate the population of each group to make a single value of pixel intensity.

    Returns:
        Float[th.Tensor, "28 28"]: Image tensor of shape (28, 28)
    """
    x = x.float()
    Populations, *feature_shape = x.shape
    device = x.device
    img = th.zeros(feature_shape).to(device)
    for j in range(Populations):
        img += j / (Populations - 1) * x[j]
    return img


if __name__ == "__main__":
    from torchvision.datasets import MNIST  # type: ignore

    mnist = MNIST("../", download=True, transform=encode_data)
    y = mnist[0][0]
    pdb.set_trace()
