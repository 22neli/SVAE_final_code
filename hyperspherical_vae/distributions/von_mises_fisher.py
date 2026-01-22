import math
import torch
from torch.distributions.kl import register_kl

from hyperspherical_vae.ops.ive import ive, ive_fraction_approx, ive_fraction_approx2
from hyperspherical_vae.distributions.hyperspherical_uniform import HypersphericalUniform


class VonMisesFisher(torch.distributions.Distribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        # option 1:
        return self.loc * (
            ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale)
        )
        # option 2:
        # return self.loc * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # option 3:
        # return self.loc * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None, k=1):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = (torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1))).to(self.device)
        self.k = k

        # --- instrumentation (for logging) ---
        self.last_accept_rate = None
        self.last_mean_attempts = None
        self.last_num_proposals = None
        self.last_num_accepted = None

        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

        w = self.__sample_w3(shape=shape) if self.__m == 3 else self.__sample_w_rej(shape=shape)

        v = (
            torch.distributions.Normal(0, 1)
            .sample(shape + torch.Size(self.loc.shape))
            .to(self.device)
            .transpose(0, -1)[1:]
        ).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self.__householder_rotation(x)

        return z.type(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        self.__w = (
            1
            + torch.stack([torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0).logsumexp(0)
            / self.scale
        )
        return self.__w

    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(
            torch.max(
                torch.tensor([0.0], dtype=self.dtype, device=self.device),
                self.scale - 10,
            ),
            torch.tensor([1.0], dtype=self.dtype, device=self.device),
        )
        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, k=self.k)
        return self.__w

    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        """
        Return the index of the first True (>0) element along `dim`.
        If no True exists along that dimension, return `invalid_val`.
        """
        mask = x > 0
        any_ = mask.any(dim=dim)
        idx = mask.float().argmax(dim=dim)  # <-- FIX: respect `dim`
        idx = torch.where(any_, idx, torch.full_like(idx, invalid_val))
        return idx

    # where the accept/reject happens
    def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
        """
        Vectorized rejection sampler.
        Proposes `k` candidates per active row per iteration until all rows accepted.

        Instrumentation:
          - last_num_proposals: total proposed candidates across all iterations
          - last_num_accepted: number of accepted samples (rows)
          - last_accept_rate: accepted / proposals
          - last_mean_attempts: proposals / accepted (unbiased)
        """
        # flatten batch to (N, 1) rows
        b, a, d = [
            e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
            for e in (b, a, d)
        ]

        w = torch.zeros_like(b, device=self.device)
        e = torch.zeros_like(b, device=self.device)
        bool_mask = torch.ones_like(b, dtype=torch.bool, device=self.device)  # True = still needs accept

        # --- instrumentation counters ---
        total_proposals = 0
        total_accepted = 0

        sample_shape = torch.Size([b.shape[0], k])
        out_shape = shape + torch.Size(self.scale.shape)

        con = torch.tensor((self.__m - 1) / 2, dtype=torch.float64, device=self.device)

        while bool_mask.any():
            active = bool_mask.squeeze(1)               # (N,)
            n_active = int(active.sum().item())
            total_proposals += n_active * k

            e_mat = (
                torch.distributions.Beta(con, con)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            u = (
                torch.distributions.Uniform(eps, 1 - eps)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            w_mat = (1 - (1 + b) * e_mat) / (1 - (1 - b) * e_mat)
            t = (2 * a * b) / (1 - (1 - b) * e_mat)

            accept_mat = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
            accept_idx = self.first_nonzero(accept_mat, dim=-1, invalid_val=-1).unsqueeze(1)  # (N,1)

            reject = accept_idx < 0     # (N,1)
            accept = ~reject            # (N,1)

            # pick first accepted proposal (or dummy index 0 if none)
            accept_idx_clamped = accept_idx.clamp(min=0)
            w_pick = w_mat.gather(1, accept_idx_clamped)
            e_pick = e_mat.gather(1, accept_idx_clamped)

            # --- FIX: update using old mask, and advance the active set correctly ---
            bool_mask_old = bool_mask
            accepted_rows = bool_mask_old & accept

            w[accepted_rows] = w_pick[accepted_rows]
            e[accepted_rows] = e_pick[accepted_rows]

            accepted_now = int(accepted_rows.sum().item())
            total_accepted += accepted_now

            # still active iff it was active and it rejected this round
            bool_mask = bool_mask_old & reject

        # --- save instrumentation on self ---
        self.last_num_proposals = total_proposals
        self.last_num_accepted = total_accepted
        self.last_accept_rate = (total_accepted / float(total_proposals)) if total_proposals > 0 else None
        # unbiased mean attempts (proposals per accepted sample)
        self.last_mean_attempts = (total_proposals / float(total_accepted)) if total_accepted > 0 else None

        return e.reshape(out_shape), w.reshape(out_shape)

    def __householder_rotation(self, x):
        u = self.__e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def entropy(self):
        # option 1:
        output = (
            -self.scale
            * ive(self.__m / 2, self.scale)
            / ive((self.__m / 2) - 1, self.scale)
        )
        # option 2:
        # output = - self.scale * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # option 3:
        # output = - self.scale * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)
        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = -(
            (self.__m / 2 - 1) * torch.log(self.scale)
            - (self.__m / 2) * math.log(2 * math.pi)
            - (self.scale + torch.log(ive(self.__m / 2 - 1, self.scale)))
        )
        return output.view(*(output.shape[:-1]))


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return -vmf.entropy() + hyu.entropy()
