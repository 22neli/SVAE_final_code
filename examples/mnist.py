import os
import csv
import random
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


# Dataset helpers (MNIST + FashionMNIST)

def get_dataset(name, root="./data", train=True):
    name = str(name).lower()
    tfm = transforms.ToTensor()

    if name == "mnist":
        return datasets.MNIST(root, train=train, download=True, transform=tfm)
    elif name in ["fashionmnist", "fashion-mnist", "fmnist"]:
        return datasets.FashionMNIST(root, train=train, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def make_loader(ds, batch_size=64, shuffle=False, generator=None):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )



# Reproducibility helpers

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Data loaders (rebuilt per run_one)

train_loader = None
test_loader = None
fixed_x_mb = None



# Model definition

class ModelVAE(nn.Module):
    def __init__(self, h_dim, z_dim, activation=F.relu, distribution="normal", kappa_clip=None, kappa_temp=1.0):
        """
        distribution: "normal" or "vmf"
        kappa_clip: None or float (only used for vMF; clamps κ)
        kappa_temp: float (only used for vMF; temperature scaling on κ via division)
        """
        super().__init__()
        self.z_dim = z_dim
        self.activation = activation
        self.distribution = distribution
        self.kappa_clip = kappa_clip
        self.kappa_temp = float(kappa_temp) if kappa_temp is not None else 1.0

        # Encoder: 784 -> 2h -> h
        self.fc_e0 = nn.Linear(784, h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        if distribution == "normal":
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, z_dim)
        elif distribution == "vmf":
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)  # κ
        else:
            raise NotImplementedError

        # Decoder: z -> h -> 2h -> 784
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 784)

    def encode(self, x):
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))

        if self.distribution == "normal":
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x)) 
        else:  # vmf
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)

            z_var = F.softplus(self.fc_var(x)) + 1.0  # κ (avoid collapse)

            # κ temperature scaling
            T = max(1e-8, float(self.kappa_temp))
            z_var = z_var / T

            # κ-clipping intervention 
            if self.kappa_clip is not None:
                z_var = torch.clamp(z_var, max=float(self.kappa_clip))

        return z_mean, z_var

    def decode(self, z):
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        return self.fc_logits(x)

    def reparameterize(self, z_mean, z_var):
        if self.distribution == "normal":
            q_z = torch.distributions.Normal(z_mean, z_var)
            p_z = torch.distributions.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        else:
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1, validate_args=False)
        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_logits = self.decode(z)
        return (z_mean, z_var), (q_z, p_z), z, x_logits



# Log-likelihood (MC estimate)

def log_likelihood(model, x, n=10):
    z_mean, z_var = model.encode(x.reshape(-1, 784))
    q_z, p_z = model.reparameterize(z_mean, z_var)

    z = q_z.rsample(torch.Size([n]))  # [n,B,D]
    x_logits = model.decode(z)

    log_p_z = p_z.log_prob(z)
    if model.distribution == "normal":
        log_p_z = log_p_z.sum(-1)

    log_p_x_z = -nn.BCEWithLogitsLoss(reduction="none")(
        x_logits, x.reshape(-1, 784).repeat((n, 1, 1))
    ).sum(-1)

    log_q_z_x = q_z.log_prob(z)
    if model.distribution == "normal":
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()


# Δ metric (posterior resampling sensitivity)

@torch.no_grad()
def latent_sensitivity_delta(model, x_mb, n_pairs=16, use_sigmoid=True, return_details=True):
    model.eval()
    device = next(model.parameters()).device
    x_mb = x_mb.to(device)
    x = x_mb.reshape(-1, 784) if x_mb.dim() > 2 else x_mb

    z_mean, z_var = model.encode(x)
    q_z, _ = model.reparameterize(z_mean, z_var)

    per_x_pairs = []
    for _ in range(n_pairs):
        z1 = q_z.rsample()
        z2 = q_z.rsample()
        x1 = model.decode(z1)
        x2 = model.decode(z2)

        if use_sigmoid:
            x1 = torch.sigmoid(x1)
            x2 = torch.sigmoid(x2)

        per_x_pairs.append((x1 - x2).abs().mean(dim=1))  # [B]

    per_x = torch.stack(per_x_pairs, dim=0).mean(dim=0)
    delta_mean = float(per_x.mean().item())

    if not return_details:
        return delta_mean

    per_x_np = per_x.detach().cpu().numpy()

    spread_proxy = None
    try:
        spread_proxy = z_var.detach().mean(dim=1).cpu().numpy()
    except Exception:
        pass

    corr = np.nan
    if spread_proxy is not None and np.std(spread_proxy) > 0 and np.std(per_x_np) > 0:
        corr = float(np.corrcoef(per_x_np, spread_proxy)[0, 1])

    return delta_mean, {
        "delta_mean": delta_mean,
        "delta_median": float(np.median(per_x_np)),
        "delta_p90": float(np.percentile(per_x_np, 90)),
        "spread_proxy_mean": float(np.mean(spread_proxy)) if spread_proxy is not None else np.nan,
        "corr(delta, spread_proxy)": corr,
    }


# Train (returns epoch stats)

def train_one_epoch(model, optimizer):
    model.train()

    out = {
        "accept_rate_mean": np.nan,
        "attempts_mean": np.nan,
        "attempts_p90": np.nan,
        "attempts_max": np.nan,
        "kappa_obs_mean": np.nan,
        "kappa_obs_p90": np.nan,
        "kappa_obs_max": np.nan,
        "accept_est_from_attempts": np.nan,
        "accept_gap": np.nan,
        "train_epoch_seconds": np.nan,
        "train_seconds_per_batch": np.nan,
    }

    vmf_accept = []
    vmf_attempts = []
    vmf_kmean = []
    vmf_kmax = []
    vmf_kp90 = []

    t0 = time.perf_counter()
    n_batches = 0

    for x_mb, _ in train_loader:
        n_batches += 1
        optimizer.zero_grad()

        # dynamic binarization (uses current torch RNG state)
        x_mb = (x_mb > torch.rand_like(x_mb)).float()

        _, (q_z, p_z), _, x_logits = model(x_mb.reshape(-1, 784))

        if model.distribution == "vmf":
            ar = getattr(q_z, "last_accept_rate", None)
            ma = getattr(q_z, "last_mean_attempts", None)

            if hasattr(q_z, "scale"):
                k = q_z.scale.detach()
                vmf_kmean.append(float(k.mean().item()))
                vmf_kmax.append(float(k.max().item()))
                vmf_kp90.append(float(torch.quantile(k.reshape(-1), 0.90).item()))

            if ar is not None:
                vmf_accept.append(float(ar))
            if ma is not None:
                vmf_attempts.append(float(ma))

        loss_recon = nn.BCEWithLogitsLoss(reduction="none")(
            x_logits, x_mb.reshape(-1, 784)
        ).sum(-1).mean()

        if model.distribution == "normal":
            loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        else:
            loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        loss = loss_recon + loss_kl
        loss.backward()
        optimizer.step()

    dt = time.perf_counter() - t0
    out["train_epoch_seconds"] = float(dt)
    out["train_seconds_per_batch"] = float(dt / max(1, n_batches))

    if model.distribution == "vmf" and len(vmf_attempts) > 0:
        attempts = np.array(vmf_attempts, dtype=float)
        out["attempts_mean"] = float(np.mean(attempts))
        out["attempts_p90"] = float(np.percentile(attempts, 90))
        out["attempts_max"] = float(np.max(attempts))

        if len(vmf_accept) > 0:
            out["accept_rate_mean"] = float(np.mean(np.array(vmf_accept, dtype=float)))

        if len(vmf_kmean) > 0:
            out["kappa_obs_mean"] = float(np.mean(np.array(vmf_kmean, dtype=float)))
        if len(vmf_kp90) > 0:
            out["kappa_obs_p90"] = float(np.mean(np.array(vmf_kp90, dtype=float)))
        if len(vmf_kmax) > 0:
            out["kappa_obs_max"] = float(np.max(np.array(vmf_kmax, dtype=float)))

        if np.isfinite(out["attempts_mean"]) and out["attempts_mean"] > 0:
            out["accept_est_from_attempts"] = float(1.0 / out["attempts_mean"])
        if np.isfinite(out["accept_rate_mean"]) and np.isfinite(out["accept_est_from_attempts"]):
            out["accept_gap"] = float(out["accept_rate_mean"] - out["accept_est_from_attempts"])

    return out



# Test (returns epoch metrics)

def test_one_epoch(model, test_seed=0):
    model.eval()
    agg = defaultdict(list)

    device = next(model.parameters()).device
    test_g = torch.Generator(device="cpu")
    test_g.manual_seed(test_seed)

    for x_mb, _ in test_loader:
        r = torch.rand(x_mb.shape, generator=test_g)
        x_mb = (x_mb > r).float().to(device)

        _, (q_z, p_z), _, x_logits = model(x_mb.reshape(-1, 784))

        recon = nn.BCEWithLogitsLoss(reduction="none")(x_logits, x_mb.reshape(-1, 784)).sum(-1).mean()
        agg["recon loss"].append(float(recon.detach().cpu().item()))

        if model.distribution == "normal":
            kl = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        else:
            kl = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        agg["KL"].append(float(kl.detach().cpu().item()))

        agg["ELBO"].append(-agg["recon loss"][-1] - agg["KL"][-1])

        ll = log_likelihood(model, x_mb)
        agg["LL"].append(float(ll.detach().cpu().item()))

    # Δ on fixed batch (built per dataset in run_one)
    delta_sig, info_sig = latent_sensitivity_delta(model, fixed_x_mb, n_pairs=16, use_sigmoid=True, return_details=True)
    delta_log, _ = latent_sensitivity_delta(model, fixed_x_mb, n_pairs=16, use_sigmoid=False, return_details=True)

    metrics = {k: float(np.mean(v)) for k, v in agg.items()}

    z_eff = float(model.z_dim)
    metrics["KL_per_dim"] = float(metrics["KL"] / max(1e-12, z_eff))

    metrics.update(
        {
            "Delta_sigmoid": float(delta_sig),
            "Delta_logits": float(delta_log),
            "Delta_sig_median": float(info_sig["delta_median"]),
            "Delta_sig_p90": float(info_sig["delta_p90"]),
            "spread_proxy_mean": float(info_sig["spread_proxy_mean"]),
            "corr_delta_spread_sigmoid": float(info_sig["corr(delta, spread_proxy)"]),
        }
    )
    return metrics



# Sweep runner

def run_one(config, out_csv):
    """
    config keys:
      - model_type: "normal" or "vmf"
      - z_dim: int (base latent dim; for vmf we use z_dim+1)
      - seed: int
      - kappa_clip: None/50/100 (vmf only)
      - kappa_temp: float (vmf only)
      - epochs: int
      - dataset: "mnist" or "fashionmnist" (or "fmnist")
    """
    set_seed(config["seed"])
    dataset_name = config.get("dataset", "mnist").lower()

    global train_loader, test_loader, fixed_x_mb

    # deterministic generator for train shuffle
    g = torch.Generator()
    g.manual_seed(config["seed"])

    # build datasets
    train_ds = get_dataset(dataset_name, root="./data", train=True)
    test_ds = get_dataset(dataset_name, root="./data", train=False)

    # loaders
    train_loader = make_loader(train_ds, batch_size=64, shuffle=True, generator=g)
    test_loader = make_loader(test_ds, batch_size=64, shuffle=False)

    # fixed batch for Δ (rebuild per dataset; deterministic)
    fixed_x_mb, _ = next(iter(test_loader))
    torch.manual_seed(0)
    fixed_x_mb = (fixed_x_mb > torch.rand_like(fixed_x_mb)).float()

    H_DIM = 128
    z_dim = int(config["z_dim"])
    epochs = int(config["epochs"])

    if config["model_type"] == "normal":
        model = ModelVAE(h_dim=H_DIM, z_dim=z_dim, distribution="normal")
    else:
        model = ModelVAE(
            h_dim=H_DIM,
            z_dim=z_dim + 1,  # our vMF uses Z_DIM+1
            distribution="vmf",
            kappa_clip=config.get("kappa_clip", None),
            kappa_temp=config.get("kappa_temp", 1.0),
        )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    fieldnames = [
        "timestamp",
        "model_type",
        "seed",
        "dataset",
        "z_dim",
        "z_dim_effective",
        "kappa_clip",
        "kappa_temp",
        "epoch",
        # train stats
        "accept_rate_mean",
        "attempts_mean",
        "attempts_p90",
        "attempts_max",
        "kappa_obs_mean",
        "kappa_obs_p90",
        "kappa_obs_max",
        "accept_est_from_attempts",
        "accept_gap",
        "train_epoch_seconds",
        "train_seconds_per_batch",
        # test stats
        "recon loss",
        "KL",
        "KL_per_dim",
        "ELBO",
        "LL",
        "Delta_sigmoid",
        "Delta_logits",
        "Delta_sig_median",
        "Delta_sig_p90",
        "spread_proxy_mean",
        "corr_delta_spread_sigmoid",
    ]

    z_eff = (z_dim if config["model_type"] == "normal" else z_dim + 1)

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, optimizer)
        test_stats = test_one_epoch(model, test_seed=0)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_type": config["model_type"],
            "seed": config["seed"],
            "dataset": dataset_name,
            "z_dim": z_dim,
            "z_dim_effective": z_eff,
            "kappa_clip": config.get("kappa_clip", None),
            "kappa_temp": config.get("kappa_temp", 1.0),
            "epoch": epoch,
            **train_stats,
            **test_stats,
        }

        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)

        if config["model_type"] == "vmf":
            print(
                f"[vmf] ds={dataset_name} seed={config['seed']} z={z_dim} (eff={z_eff}) "
                f"kclip={config.get('kappa_clip')} T={config.get('kappa_temp', 1.0)} "
                f"ep={epoch} ELBO={row['ELBO']:.2f} KL={row['KL']:.2f} KL/d={row['KL_per_dim']:.3f} "
                f"acc={row['accept_rate_mean']:.3f} "
                f"1/att={row['accept_est_from_attempts']:.3f} gap={row['accept_gap']:.3e} "
                f"att(mean/p90/max)={row['attempts_mean']:.2f}/{row['attempts_p90']:.2f}/{row['attempts_max']:.2f} "
                f"k(mean/p90/max)={row['kappa_obs_mean']:.1f}/{row['kappa_obs_p90']:.1f}/{row['kappa_obs_max']:.1f} "
                f"time={row['train_epoch_seconds']:.2f}s ({row['train_seconds_per_batch']:.3f}s/b) "
                f"Δsig={row['Delta_sigmoid']:.4f}"
            )
        else:
            print(
                f"[normal] ds={dataset_name} seed={config['seed']} z={z_dim} "
                f"ep={epoch} ELBO={row['ELBO']:.2f} KL={row['KL']:.2f} KL/d={row['KL_per_dim']:.3f} "
                f"time={row['train_epoch_seconds']:.2f}s ({row['train_seconds_per_batch']:.3f}s/b) "
                f"Δsig={row['Delta_sigmoid']:.4f}"
            )


def run_sweep():
    out_csv = "results_svae_sweep.csv"

    datasets_to_run = ["mnist", "fashionmnist"]  # run both
    z_sweep = [5, 10, 20]
    seeds = [0, 1]
    epochs = 20

    for ds in datasets_to_run:
        # 1) Baseline sweeps: Normal + vMF (no clipping)
        for seed in seeds:
            for z in z_sweep:
                run_one({"model_type": "normal", "dataset": ds, "z_dim": z, "seed": seed, "epochs": epochs}, out_csv)
                run_one(
                    {"model_type": "vmf", "dataset": ds, "z_dim": z, "seed": seed, "epochs": epochs,
                     "kappa_clip": None, "kappa_temp": 1.0},
                    out_csv,
                )

        # 2) κ-clipping ablation on "stress" setting (z=20)
        stress_z = 20
        for seed in seeds:
            for kclip in [50, 100]:
                run_one(
                    {"model_type": "vmf", "dataset": ds, "z_dim": stress_z, "seed": seed, "epochs": epochs,
                     "kappa_clip": kclip, "kappa_temp": 1.0},
                    out_csv,
                )

        # 3) κ-temperature scaling ablation on stress setting (z=20)
        for seed in seeds:
            for T in [0.5, 1.0, 2.0]:
                run_one(
                    {"model_type": "vmf", "dataset": ds, "z_dim": stress_z, "seed": seed, "epochs": epochs,
                     "kappa_clip": None, "kappa_temp": T},
                    out_csv,
                )

    print(f"\nDone. Saved: {out_csv}")


if __name__ == "__main__":
    run_sweep()
