import enum
from importlib.metadata import distribution
from itertools import count
from math import log
import pdb
import trace
from turtle import forward, st
import torch as th
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from jaxtyping import UInt8, Float, Float64, Int, Bool
import wandb
from tqdm.auto import tqdm

counter = count(1)
state_history = []


class HMM(th.nn.Module):
    log_likelihood_afferent: Float64[th.Tensor, "in_features out_features"]
    "Probability of the input spike given the latent variable, shape of: (in_features, out_features)"
    log_likelihood_lateral: Float64[th.Tensor, "out_features out_features"]
    "Probability of the latent variable given the previous latent variable, shape of: (out_features, out_features)"
    log_prior: Float64[th.Tensor, "out_features"]
    "Marginal probability of the latent variable, shape of: (out_features, )"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 1e-1,
        populations: int = 2,
        tau: float = 10.0,
        refractory_period: int = 5,
        num_paths: int = 10,
        num_steps: int = 50,
        dtype: th.dtype = th.float64,
    ) -> None:
        super(HMM, self).__init__()
        self.in_features = in_features
        "Number of input features"
        self.out_features = out_features
        "Number of output features"
        self.learning_rate = learning_rate
        self.populations = populations
        "Number of populations for population coding"
        self.tau = tau
        "Time constant for the membrane potential"
        self.refractory_period = refractory_period
        "Refractory period for the membrane potential"
        self.num_paths = num_paths
        "Number of paths to get the mean of the importance weights"
        self.num_steps = num_steps
        "Number of time steps"
        self.ltp_constant = 7
        "Constant for the LTP rule"
        self.inverse_lr_decay = 1
        "To garantee the convergence of the algorithm"

        log_likelihood_afferent = (
            th.rand(in_features, out_features, dtype=dtype) * -1 - 1
        )
        log_likelihood_lateral = (
            th.rand(out_features, out_features, dtype=dtype) * -1 - 1
        )

        log_prior = th.ones(out_features, dtype=dtype) * -1
        self.register_buffer("log_likelihood_afferent", log_likelihood_afferent)
        self.register_buffer("log_likelihood_lateral", log_likelihood_lateral)
        self.register_buffer("log_prior", log_prior)

        self.normalize_probs()

        self.trace_afferent: Float64[th.Tensor, "Batch in_features"]
        """Float64[th.Tensor, "Batch in_features"]"""
        self.trace_lateral: Float64[th.Tensor, "Batch out_features"]
        """Float64[th.Tensor, "Batch out_features"]"""
        self.trace_post: Float64[th.Tensor, "Batch out_features"]
        """Float64[th.Tensor, "Batch out_features"]"""
        self.dw_afferent: Float64[th.Tensor, "Batch in_features out_features"]
        """Float64[th.Tensor, "Batch in_features out_features"]"""
        self.dw_lateral: Float64[th.Tensor, "Batch out_features out_features"]
        """Float64[th.Tensor, "Batch out_features out_features"]"""
        self.db: Float64[th.Tensor, "Batch out_features"]
        """Float64[th.Tensor, "Batch out_features"]"""

    def reset_trace(self, batch: int, device: th.device) -> None:
        self.trace_afferent = th.zeros(
            (batch, self.in_features), dtype=th.float64, device=device
        )
        self.trace_lateral = th.zeros(
            (batch, self.out_features), dtype=th.float64, device=device
        )
        self.trace_post = th.zeros(
            (batch, self.out_features), dtype=th.float64, device=device
        )
        self.dw_afferent = th.zeros(
            (batch, self.in_features, self.out_features),
            dtype=th.float64,
            device=device,
        )
        self.dw_lateral = th.zeros(
            (batch, self.out_features, self.out_features),
            dtype=th.float64,
            device=device,
        )
        self.db = th.zeros((batch, self.out_features), dtype=th.float64, device=device)

    # trace_table = []

    def save_trace(
        self,
        afferent: Bool[th.Tensor, "Batch in_features"],
        lateral_prev: Bool[th.Tensor, "Batch out_features"],
        lateral_current: Bool[th.Tensor, "Batch out_features"],
    ):
        ##############################################
        # Save the afferent stdp
        pre_post_afferent = (-self.log_likelihood_afferent).exp() * (
            self.trace_afferent.unsqueeze(2) @ lateral_current.double().unsqueeze(1)
        ) - 1
        # post_pre_afferent = afferent.double().unsqueeze(2) @ (
        #     self.trace_lateral * (1 - 1 / self.tau) + lateral_current.double()
        # ).unsqueeze(1)
        self.dw_afferent += pre_post_afferent  # - post_pre_afferent
        ##############################################
        # Save the lateral stdp
        pre_post_lateral = (-self.log_likelihood_lateral).exp() * (
            self.trace_lateral.unsqueeze(2) @ lateral_current.double().unsqueeze(1)
        ) - 1
        # post_pre_lateral = lateral_prev.double().unsqueeze(2) @ (
        #     self.trace_lateral + lateral_current.double()
        # ).unsqueeze(1)
        self.dw_lateral += pre_post_lateral  # - post_pre_lateral
        ##############################################
        # Save the prior stdp
        self.db += (1 / self.out_features - 1) * lateral_current.double() + (
            lateral_current.logical_not().double()
        ) / self.out_features
        ##############################################
        # Update the traces
        self.trace_afferent += -self.trace_afferent / self.tau + afferent.double()
        self.trace_lateral += -self.trace_lateral / self.tau + lateral_current.double()

        ##############################################

    def accept_stdp(self, index: int) -> None:
        scaled_lr = self.learning_rate / self.inverse_lr_decay
        self.log_likelihood_afferent += scaled_lr * self.dw_afferent[index]
        self.log_likelihood_lateral += scaled_lr * self.dw_lateral[index]
        self.log_prior += scaled_lr + self.db[index]

        self.inverse_lr_decay += 1
        self.normalize_probs()

        wandb.log(
            {
                f"lateral_sum_{i}": k
                for i, k in enumerate(self.log_likelihood_lateral.exp().sum(dim=0))
            }
        )

    def to_epsp(self, x: Bool[th.Tensor, "Batch Num_steps Num_Population*28*28"]):
        x_new = th.zeros_like(x, dtype=th.float64, device=x.device)
        for t in range(self.num_steps):
            epsp = x[:, t].double()  # (Batch, Num_Population*28*28)
            for tp in range(t, self.num_steps):
                x_new[:, tp] += epsp
                epsp *= 1 - 1 / self.tau
        return x_new

    def forward_train(self, x: Bool[th.Tensor, "Batch Num_steps Num_Population*28*28"]):
        assert len(x.shape) == 3, (
            "Input shape must be (Batch, Num_steps, Num_Population*28*28)"
        )
        x = x.bool()
        device = x.device
        batch_size, num_steps, in_features = x.shape

        # Begin rejection sampling
        for trial in tqdm(counter, leave=False, desc="Rejection sampling trials"):
            sample_count = batch_size * self.num_paths
            potentials = th.zeros(sample_count, self.out_features, device=device)

            prev_states = th.zeros(
                sample_count,
                self.out_features,
                device=device,
                dtype=th.bool,
            )  # (Batch * Paths, out_features).

            refractory_remains = th.zeros_like(potentials, device=device, dtype=th.int)

            log_importance_weights = th.zeros(
                sample_count, device=device, dtype=th.float64
            )

            # Compute p((x_m)_i|(s_{m-1})_j), where marginal_ll[i, j] = p((x_m)_i|(s_{m-1})_j).
            # Transpose the log_likelihood_lateral, as it is left-to-right transform.
            # I.e. it calculates all the paths of prev_state -> current_state -> affarent.
            marginal_ll = (
                self.log_likelihood_afferent.exp().mm(
                    self.log_likelihood_lateral.T.exp()
                )
            ).log()  # (in_features, out_features)
            self.reset_trace(sample_count, device)

            for t in range(num_steps):
                x_t = x[:, t].repeat_interleave(
                    self.num_paths, dim=0
                )  # (Batch * Paths, in_features)
                wandb.log(
                    {
                        f"lateral/potentials_{k}": potentials[0, k]
                        for k in range(self.out_features)
                    }
                )
                potentials = (
                    potentials * (1 - 1 / self.tau)
                    + x_t.double() @ self.log_likelihood_afferent
                    # +p_t @ self.log_likelihood_afferent
                    + prev_states.double() @ self.log_likelihood_lateral
                    + self.log_prior
                )  # (Batch * Paths, out_features)

                neg_energy = potentials.where(refractory_remains == 0, -th.inf)
                posteriors = neg_energy.softmax(dim=1)  # (Batch * Paths, out_features)
                """
                Assume that there is always exactly one spike at each time step, following discussed circuit homeostasis in the paper.
                Sample latent states multiple times to estimate the mean of importance weights.
                """
                states = Bernoulli(posteriors).sample()  # (Batch * Paths, out_features)
                refractory_remains = (
                    refractory_remains - 1 + states * self.refractory_period
                ).clamp(min=0)
                wandb.log(
                    {
                        f"lateral/spikes_{k}": states.sum(dim=0)[k]
                        for k in range(self.out_features)
                    }
                )

                if t != 0:
                    instantaneous_input_ll = th.zeros(
                        sample_count, device=device, dtype=th.float64
                    )
                    for i in range(sample_count):
                        # Compute the sum of the log likelihoods of the input spikes given the latent states.
                        instantaneous_input_ll[i] = (
                            x_t[i].double().unsqueeze(0) @ marginal_ll @ prev_states[i]
                        )
                        # instantaneous_input_ll[i] = marginal_ll[x_t[i]][
                        #     :, prev_states[i].int().argmax(dim=0)
                        # ].sum()
                    log_importance_weights += instantaneous_input_ll

                self.save_trace(x_t, prev_states, states)

                for i in range(sample_count):
                    for j in range(self.out_features):
                        if states[i, j]:
                            state_history.append([j])

                prev_states = states

            # Acceptance step
            # Normalize the importance weights
            log_importance_weights = log_importance_weights[
                :: self.num_paths
            ] - log_importance_weights.view(batch_size, self.num_paths).logsumexp(
                dim=1, keepdim=False
            )
            acceptance = Bernoulli(log_importance_weights.exp().clamp(max=1)).sample()

            wandb.log(
                {
                    "rejection sampling/batch size": batch_size,
                    "rejection sampling/acceptance": acceptance.sum(),
                    "rejection sampling/log importance weights": log_importance_weights.mean(),
                    "state distribution": wandb.plot.histogram(
                        wandb.Table(columns=["state"], data=state_history),
                        value="state",
                        title="State distribution",
                    ),
                },
            )
            for i in range(batch_size):
                # acceptance[i] = 1
                if acceptance[i]:
                    self.accept_stdp(i * self.num_paths)
                    batch_size -= 1
                else:
                    continue

            if batch_size == 0:
                break
            x = x[acceptance.bool().logical_not()]
        return states

    def forward_gen(
        self, batch_size: int
    ) -> Bool[th.Tensor, "Batch Num_steps Num_Population*28*28"]:
        # hidden_states = th.zeros(
        #     batch_size,
        #     self.out_features,
        #     dtype=th.bool,
        #     device=self.log_likelihood_afferent.device,
        # )  # (Batch * Paths, out_features).
        hidden_states = Categorical(self.log_prior.softmax(dim=0)).sample((batch_size,))
        hidden_states = F.one_hot(
            hidden_states, self.out_features
        ).bool()  # (Batch, out_features)

        x = th.zeros(
            batch_size,
            self.num_steps,
            self.in_features,
            dtype=th.bool,
            device=self.log_likelihood_afferent.device,
        )

        for t in range(self.num_steps - 1, -1, -1):
            state_distribution = (
                self.log_likelihood_lateral.exp() @ hidden_states.T.double()
            ).T.log()  # (Batch, out_features)
            state_distribution += self.log_prior
            hidden_states = Categorical(
                state_distribution.softmax(dim=1)
            ).sample()  # (Batch)

            x[:, t, :] = (
                Bernoulli(self.log_likelihood_afferent[:, hidden_states].exp())
                .sample()
                .T
            )  # (Batch, in_features)

            hidden_states = F.one_hot(
                hidden_states, self.out_features
            ).bool()  # (Batch, out_features)

        return x

    def forward(
        self,
        x: Bool[th.Tensor, "Batch Num_steps Num_Population*28*28"] | None = None,
        batch_size: int = 1,
    ):
        if x is not None:
            return self.forward_train(x)
        else:
            return self.forward_gen(batch_size)

    def normalize_probs(self) -> None:
        """Normalize over in_features, to satisfy the constraint that the sum of the probs to each output neuron is 1.
        (the weights is log prob of the input spike given the latent variable)"""
        self.log_likelihood_afferent.clamp_(min=-self.ltp_constant, max=0)
        population_form = self.log_likelihood_afferent.view(
            self.populations, -1, self.out_features
        )
        self.log_likelihood_afferent = (
            population_form - population_form.logsumexp(dim=0, keepdim=True)
        ).view(*self.log_likelihood_afferent.shape)
        self.log_likelihood_lateral.clamp_(min=-self.ltp_constant, max=0)
        self.log_likelihood_lateral -= self.log_likelihood_lateral.logsumexp(
            dim=1, keepdim=True
        )
        self.log_prior.clamp_(min=-self.ltp_constant, max=0)
        self.log_prior -= self.log_prior.logsumexp(dim=0)


if __name__ == "__main__":
    import wandb

    wandb.init(mode="offline")
    hmm = HMM(784 * 2, 10)
    x = th.Tensor(
        [[1, 0], [0, 1], [0, 0], [0, 0], [1, 0], [0, 0], [0, 1]] + [[0, 0]] * 43
    ).unsqueeze(0)
    print(x.shape)
    print(hmm.to_epsp(x))
    pdb.set_trace()
