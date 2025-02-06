import trace
from turtle import forward, st
import torch as th
import torch.nn.functional as F
from jaxtyping import UInt8, Float, Float64, Int, Bool


class HMM(th.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 1e-3,
        sigma: int = 10,
        tau: float = 5.0,
        num_paths: int = 10,
    ) -> None:
        super(HMM, self).__init__()
        self.in_features = in_features
        "Number of input features"
        self.out_features = out_features
        "Number of output features"
        self.learning_rate = learning_rate
        self.sigma = sigma
        "Time window for membrane potential and STDP"
        self.tau = tau
        "Time constant for the membrane potential"
        self.num_paths = num_paths
        "Number of paths to get the mean of the importance weights"
        self.ltp_constant = 1

        self.log_likelihood_afferent = th.nn.Parameter(
            th.rand(in_features, out_features) * -1 - 1, requires_grad=False
        )
        self.log_likelihood_lateral = th.nn.Parameter(
            th.rand(out_features, out_features) * -1 - 1, requires_grad=False
        )
        self.log_prior = th.nn.Parameter(
            th.rand(out_features) * -1 - 1, requires_grad=False
        )
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
        self.log_likelihood_afferent += self.dw_afferent[index]
        self.log_likelihood_lateral += self.dw_lateral[index]
        self.log_prior += self.db[index]

        self.normalize_probs()

    def forward(self, x: Bool[th.Tensor, "Batch Num_steps Num_Population*28*28"]):
        assert len(x.shape) == 3, (
            "Input shape must be (Batch, Num_steps, Num_Population*28*28)"
        )
        x = x.bool()
        batch, num_steps, in_features = x.shape

        # Begin rejection sampling
        while True:
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
                state_candidates = th.distributions.Categorical(posteriors).sample(
                    (self.num_paths,)
                )  # (Batch, Paths)
                assert state_candidates.shape == (batch, self.num_paths)
                state_candidates = (
                    F.one_hot(state_candidates.flatten(), self.out_features)
                    .to(th.bool)
                    .view(batch, self.num_paths, self.out_features)
                )  # (Batch, Paths, out_features)
                state = state_candidates[:, 0]  # (Batch, out_features)

                instantaneous_input_ll = th.zeros(
                    batch, self.num_paths, device=x.device, dtype=th.float64
                )
                if t != 0:
                    for i in range(batch):
                        for j in range(self.num_paths):
                            # Compute the sum of the log likelihoods of the input spikes given the latent states.
                            instantaneous_input_ll[i][j] = marginal_ll[x_t[i]][
                                :, prev_state_candidates[i, j].argmax(dim=0)
                            ].sum()
                instantaneous_input_ll = instantaneous_input_ll[
                    :, 0
                ] - instantaneous_input_ll.logsumexp(dim=1)
                log_importance_weights += instantaneous_input_ll

                self.save_trace(x_t, prev_states, state)

                prev_states = state
                prev_state_candidates = state_candidates

            # Acceptance step
            acceptance = th.distributions.Bernoulli(
                log_importance_weights.logit()
            ).sample()
            if acceptance.sum() == batch:
                break
            else:
                for i in reversed(range(batch)):
                    if acceptance[i]:
                        self.accept_stdp(i)

                        x = x[:i]
                        batch -= 1
                    else:
                        continue

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
        population_form = self.log_likelihood_lateral.view(
            self.populations, -1, self.out_features
        )
        self.log_likelihood_lateral = (
            population_form - population_form.logsumexp(dim=0, keepdim=True)
        ).view(*self.log_likelihood_lateral.shape)
        self.log_prior -= self.log_prior.logsumexp(dim=0)
