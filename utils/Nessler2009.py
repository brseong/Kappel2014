import math
import pdb
from turtle import pd
import torch as th
from torch.nn import Module
from jaxtyping import UInt8, Float, Float64, Int
from matplotlib import pyplot as plt
import wandb


class Nessler2009(Module):
    log_likelihood: Float64[th.Tensor, "in_features out_features"]  # noqa: F722
    "Probability of the input spike given the latent variable, shape of: (populations * in_features, out_features)"
    log_prior: Float64[th.Tensor, "out_features"]  # noqa: F821
    "Marginal probability of the latent variable, shape of: (out_features, )"
    trace_pre_sigma: UInt8[th.Tensor, "Batch Timesteps in_features"]
    "Trace of LTP in presynaptic spike. type: UInt8[th.Tensor, 'Batch Timesteps in_features']"
    trace_pre_sigma2inf: UInt8[th.Tensor, "Batch Timesteps in_features"]
    "Trace of LTD in presynaptic spike. type: UInt8[th.Tensor, 'Batch Timesteps in_features']"
    trace_post_2sigma: UInt8[th.Tensor, "Batch Timesteps out_features"]
    "Trace of LTP in postsynaptic spike. type: UInt8[th.Tensor, 'Batch Timesteps out_features']"

    def __init__(
        self,
        in_features: int,
        out_features: int = 10,
        learning_rate: float = 1e-3,
        populations: int = 2,
        dtype: th.dtype = th.float64,
    ) -> None:
        super(Nessler2009, self).__init__()  # type: ignore

        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.inverse_lr_decay = 1
        "To garantee the convergence of the algorithm"
        self.populations = populations
        "Number of populations for population coding"
        self.sigma = 10
        "Time window for membrane potential and STDP"
        self.ltp_constant = 1
        "Constant for the LTP rule"
        self.dtype = dtype

        # log_likelihood = th.rand((in_features, out_features), dtype=dtype) * -1 - 1
        log_likelihood = th.full((in_features, out_features), 1, dtype=th.float64)
        # log_prior = th.rand((out_features,), dtype=dtype) * -1 - 1
        log_prior = th.full((out_features,), 1, dtype=th.float64)
        self.register_buffer("log_likelihood", log_likelihood)
        self.register_buffer("log_prior", log_prior)
        self.normalize_probs()

    def forward(
        self, x: UInt8[th.Tensor, "Batch Num_steps Num_Population*28*28"]
    ) -> int:
        assert len(x.shape) == 3, (
            "Input shape must be (Batch, Num_steps, Num_Population*28*28)"
        )
        batch, num_steps, in_features = x.shape

        # Simulating the simplified membrane potential
        potential = x.clone().bool()
        for t in range(1, self.sigma):
            potential[:, t:, :] |= x[:, :-t, :].bool()
        potential = potential.to(th.uint8)

        # To save the timings of the last spike for STDP
        self.trace_pre_sigma = th.zeros(
            (batch, num_steps, in_features), dtype=th.uint8, device=x.device
        )
        self.trace_pre_sigma2inf = th.zeros(
            (batch, num_steps, in_features), dtype=th.uint8, device=x.device
        )
        self.trace_pre_2sigma = th.zeros(
            (batch, num_steps, in_features), dtype=th.uint8, device=x.device
        )
        self.trace_post_2sigma = th.zeros(
            (batch, num_steps, self.out_features),
            dtype=th.uint8,
            device=x.device,
        )
        self.trace_post_2sigma2inf = th.zeros(
            (batch, num_steps, self.out_features),
            dtype=th.uint8,
            device=x.device,
        )
        pred = th.zeros(
            (batch, num_steps, self.out_features), dtype=th.int, device=x.device
        )
        z_total = 0
        for t in range(potential.shape[1]):
            potential_t = potential[:, t, :]  # (Batch, in_features)
            posterior = (
                potential_t.double() @ self.log_likelihood + self.log_prior
            ).softmax(dim=1)  # (Batch, out_features)
            winners = (
                th.distributions.Categorical(posterior)
                .sample()
                .view(potential_t.shape[0])
            )  # (Batch, )
            z_total += winners.sum().item()
            winners = th.nn.functional.one_hot(winners, self.out_features).to(
                th.uint8
            )  # (Batch, out_features)
            self.trace_pre_sigma[:, t : t + self.sigma, :] += x[
                :, t : t + 1, :
            ]  # (Batch, sigma, in_features). Broadcast the spike through the time dimension
            self.trace_pre_sigma2inf[:, t + self.sigma :, :] += x[:, t : t + 1, :]

            self.trace_pre_2sigma[:, t + 1 : t + 1 + 2 * self.sigma, :] += x[
                :, t : t + 1, :
            ]
            self.trace_post_2sigma[:, t + 1 : t + 1 + 2 * self.sigma, :] += (
                winners.unsqueeze(1)
            )
            self.trace_post_2sigma2inf[:, t + 1 + 2 * self.sigma :, :] += (
                winners.unsqueeze(1)
            )

            self.STDP(x[:, t, :], winners, t)
            pred[:, t, :] = winners
        return pred.sum(dim=1).argmax(dim=1)

    def STDP(
        self,
        pre_spikes: UInt8[th.Tensor, "Batch in_features"],
        post_spikes: UInt8[th.Tensor, "Batch out_features"],
        t: int,
    ) -> None:
        pre_spikes = pre_spikes.double()
        post_spikes = post_spikes.double()
        # The cases when the input neuron spikes before the latent neuron
        pre_post_ltp = self.trace_pre_sigma[:, t, :].double().unsqueeze(
            -1
        ) @ post_spikes.unsqueeze(1)  # (Batch, in_features, out_features)
        pre_post_ltd = self.trace_pre_sigma2inf[:, t, :].double().unsqueeze(
            -1
        ) @ post_spikes.unsqueeze(1)  # (Batch, in_features, out_features)

        # The case when the input neuron spikes  after the latent neuron
        post_pre = pre_spikes.unsqueeze(-1) @ self.trace_post_2sigma[
            :, t, :
        ].double().unsqueeze(1)  # (Batch, in_features, out_features)

        # The case when the input neuron does not spike after the latent neuron
        post_only = (
            # If the input neuron does not spike in the time window
            self.trace_pre_2sigma[:, t + 1 - 2 * self.sigma : t + 1, :].sum(dim=1) == 0
        ).double().unsqueeze(-1) @ (
            # If the latent neuron spikes in the time window
            self.trace_post_2sigma2inf[:, t, :]
        ).double().unsqueeze(1)

        dw = (
            (
                (self.ltp_constant * (-self.log_likelihood).exp() - 1)
                * pre_post_ltp.sum(dim=0)
                - pre_post_ltd.sum(dim=0)
                - post_pre.sum(dim=0)
                - post_only.sum(dim=0)
            )
            * self.learning_rate
            / self.inverse_lr_decay
        )  # (in_features, out_features)
        self.log_likelihood += dw

        db = (
            (
                (self.ltp_constant * (-self.log_prior).exp() - 1)
                * post_spikes.sum(dim=0)
                - (1 - post_spikes.sum(dim=0))
            )
            * self.learning_rate
            / self.inverse_lr_decay
        )
        self.log_prior += db

        wandb.log(
            {
                "dw": dw.norm().item(),
                "db": db.norm().item(),
            }
        )

        self.inverse_lr_decay += 1
        self.normalize_probs()

    def normalize_probs(self) -> None:
        """Normalize over in_features, to satisfy the constraint that the sum of the probs to each output neuron is 1.
        (the weights is log prob of the input spike given the latent variable)"""
        self.log_likelihood.clamp_(min=-7, max=0)
        population_form = self.log_likelihood.view(
            self.populations, -1, self.out_features
        )
        self.log_likelihood = (
            population_form - population_form.logsumexp(dim=0, keepdim=True)
        ).view(*self.log_likelihood.shape)
        self.log_prior.clamp_(min=-7, max=0)
        self.log_prior -= self.log_prior.logsumexp(dim=0)
