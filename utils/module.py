from itertools import count
import trace
from turtle import forward, st
import torch as th
import torch.nn.functional as F
from jaxtyping import UInt8, Float, Float64, Int, Bool
import wandb
from tqdm.auto import tqdm

counter = count(1)


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
        learning_rate: float = 1e-3,
        populations: int = 2,
        sigma: int = 10,
        tau: float = 5.0,
        num_paths: int = 10,
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
        self.sigma = sigma
        "Time window for membrane potential and STDP"
        self.tau = tau
        "Time constant for the membrane potential"
        self.num_paths = num_paths
        "Number of paths to get the mean of the importance weights"
        self.ltp_constant = 1
        self.inverse_lr_decay = 1
        "To garantee the convergence of the algorithm"

        log_likelihood_afferent = (
            th.rand(in_features, out_features, dtype=dtype) * -1 - 1
        )
        log_likelihood_lateral = (
            th.rand(out_features, out_features, dtype=dtype) * -1 - 1
        )
        log_prior = th.rand(out_features, dtype=dtype) * -1 - 1
        self.register_buffer("log_likelihood_afferent", log_likelihood_afferent)
        self.register_buffer("log_likelihood_lateral", log_likelihood_lateral)
        self.register_buffer("log_prior", log_prior)

        self.normalize_probs()

        self.trace_pre_afferent: Float64[th.Tensor, "Batch in_features"]
        """Float64[th.Tensor, "Batch in_features"]"""
        self.trace_pre_lateral: Float64[th.Tensor, "Batch out_features"]
        """Float64[th.Tensor, "Batch out_features"]"""
        self.trace_post: Float64[th.Tensor, "Batch out_features"]
        """Float64[th.Tensor, "Batch out_features"]"""
        self.dw_afferent: Float64[th.Tensor, "Batch in_features out_features"]
        """Float64[th.Tensor, "Batch in_features out_features"]"""
        self.dw_lateral: Float64[th.Tensor, "Batch out_features out_features"]
        """Float64[th.Tensor, "Batch out_features out_features"]"""
        self.db: Float64[th.Tensor, "Batch out_features"]
        """Float64[th.Tensor, "Batch out_features"]"""

    def reset_trace(self, batch: int, x: th.Tensor) -> None:
        self.trace_pre_afferent = th.zeros(
            (batch, self.in_features), dtype=th.float64, device=x.device
        )
        self.trace_pre_lateral = th.zeros(
            (batch, self.out_features), dtype=th.float64, device=x.device
        )
        self.trace_post = th.zeros(
            (batch, self.out_features), dtype=th.float64, device=x.device
        )
        self.dw_afferent = th.zeros(
            (batch, self.in_features, self.out_features),
            dtype=th.float64,
            device=x.device,
        )
        self.dw_lateral = th.zeros(
            (batch, self.out_features, self.out_features),
            dtype=th.float64,
            device=x.device,
        )
        self.db = th.zeros(
            (batch, self.out_features), dtype=th.float64, device=x.device
        )

    def save_trace(
        self,
        afferent: Bool[th.Tensor, "Batch in_features"],
        lateral_prev: Bool[th.Tensor, "Batch out_features"],
        lateral_current: Bool[th.Tensor, "Batch out_features"],
    ):
        ##############################################
        # Update the traces
        self.trace_pre_afferent += (
            -self.trace_pre_afferent / self.tau + afferent.double()
        )
        self.trace_pre_lateral += (
            -self.trace_pre_lateral / self.tau + lateral_prev.double()
        )
        self.trace_post += -self.trace_post / self.tau + lateral_current.double()

        ##############################################
        # Save the afferent stdp
        pre_post_afferent = (
            self.log_likelihood_afferent.exp()
            * (
                self.trace_pre_afferent.unsqueeze(2)
                @ lateral_current.double().unsqueeze(1)
            )
            - 1
        )
        post_pre_afferent = afferent.double().unsqueeze(2) @ self.trace_post.unsqueeze(
            1
        )
        self.dw_afferent += pre_post_afferent - post_pre_afferent
        ##############################################
        # Save the lateral stdp
        pre_post_lateral = (
            self.log_likelihood_lateral.exp()
            * (
                self.trace_pre_lateral.unsqueeze(2)
                @ lateral_current.double().unsqueeze(1)
            )
            - 1
        )
        post_pre_lateral = lateral_prev.double().unsqueeze(
            2
        ) @ self.trace_post.unsqueeze(1)
        self.dw_lateral += pre_post_lateral - post_pre_lateral
        ##############################################
        # Save the prior stdp
        self.db += self.log_prior.exp() * lateral_current.double() - (
            1 - lateral_current.double()
        )

    def accept_stdp(self, index: int) -> None:
        self.log_likelihood_afferent += (
            self.learning_rate * self.dw_afferent[index] / self.inverse_lr_decay
        )
        self.log_likelihood_lateral += (
            self.learning_rate * self.dw_lateral[index] / self.inverse_lr_decay
        )
        self.log_prior += self.learning_rate * self.db[index] / self.inverse_lr_decay

        self.inverse_lr_decay += 1
        self.normalize_probs()

    def forward(self, x: Bool[th.Tensor, "Batch Num_steps Num_Population*28*28"]):
        assert len(x.shape) == 3, (
            "Input shape must be (Batch, Num_steps, Num_Population*28*28)"
        )
        x = x.bool()
        batch, num_steps, in_features = x.shape

        # Begin rejection sampling
        for trial in tqdm(counter):
            potentials = th.zeros(batch, self.out_features, device=x.device)

            # The "0" state is the initial state: s_0.
            prev_states = F.one_hot(
                th.zeros(batch, device=x.device, dtype=th.int64), self.out_features
            ).to(th.bool)  # (Batch, out_features).
            prev_state_candidates = th.zeros(
                batch,
                self.num_paths,
                self.out_features,
                device=x.device,
                dtype=th.int64,
            )  # (Batch, Paths, out_features)

            log_importance_weights = th.zeros(batch, device=x.device, dtype=th.float64)

            # Compute p(x_m|s_{m-1}).
            marginal_ll = (
                self.log_likelihood_afferent.exp() @ self.log_likelihood_lateral.exp()
            ).log()  # (in_features, out_features)
            self.reset_trace(batch, x)

            for t in range(num_steps):
                x_t = x[:, t]
                potentials = (
                    potentials * (1 - 1 / self.tau)
                    + x_t.double() @ self.log_likelihood_afferent
                    + prev_states.double() @ self.log_likelihood_lateral
                    + self.log_prior
                )
                posteriors = potentials.softmax(dim=1)
                """
                Assume that there is always exactly one spike at each time step, following discussed circuit homeostasis in the paper.
                Sample latent states multiple times to estimate the mean of importance weights.
                """
                state_candidates = (
                    th.distributions.Categorical(posteriors).sample((self.num_paths,)).T
                )  # (Batch, Paths)
                assert state_candidates.shape == (batch, self.num_paths)
                state_candidates = (
                    F.one_hot(state_candidates.flatten(), self.out_features)
                    .to(th.bool)
                    .view(batch, self.num_paths, self.out_features)
                )  # (Batch, Paths, out_features)
                states = state_candidates[:, 0]  # (Batch, out_features)

                instantaneous_input_ll = th.zeros(
                    batch, self.num_paths, device=x.device, dtype=th.float64
                )
                if t != 0:
                    for i in range(batch):
                        for j in range(self.num_paths):
                            # Compute the sum of the log likelihoods of the input spikes given the latent states.
                            instantaneous_input_ll[i, j] = marginal_ll[x_t[i]][
                                :, prev_state_candidates[i, j].int().argmax(dim=0)
                            ].sum()
                instantaneous_input_ll = (
                    instantaneous_input_ll[:, 0]
                    - instantaneous_input_ll.logsumexp(dim=1)
                    + th.log(th.tensor(self.num_paths))
                )
                log_importance_weights += instantaneous_input_ll

                self.save_trace(x_t, prev_states, states)

                prev_states = states
                prev_state_candidates = state_candidates

            # Acceptance step
            acceptance = th.distributions.Bernoulli(
                log_importance_weights.exp().clamp(max=1)
            ).sample()

            wandb.log(
                {
                    "acceptance rate": acceptance.float().mean(),
                    "log importance weights": log_importance_weights.mean(),
                }
            )

            for i in reversed(range(batch)):
                if acceptance[i]:
                    self.accept_stdp(i)

                    x = x[:i]
                    batch -= 1
                else:
                    continue
            if batch == 0:
                break

        return states.int().argmax(dim=1)

    def normalize_probs(self) -> None:
        """Normalize over in_features, to satisfy the constraint that the sum of the probs to each output neuron is 1.
        (the weights is log prob of the input spike given the latent variable)"""
        self.log_likelihood_afferent.clamp_(min=-7, max=0)
        population_form = self.log_likelihood_afferent.view(
            self.populations, -1, self.out_features
        )
        self.log_likelihood_afferent = (
            population_form - population_form.logsumexp(dim=0, keepdim=True)
        ).view(*self.log_likelihood_afferent.shape)
        self.log_prior.clamp_(min=-7, max=0)
        self.log_likelihood_lateral.clamp_(min=-7, max=0)
        self.log_likelihood_lateral -= self.log_likelihood_lateral.logsumexp(
            dim=1, keepdim=True
        )
        self.log_prior -= self.log_prior.logsumexp(dim=0)
