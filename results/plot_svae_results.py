# plot_svae_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "results_svae_sweep.csv"
OUT_DIR = BASE_DIR / "plots_svae"
OUT_DIR.mkdir(exist_ok=True)


"""
NOTE:
We intentionally generate more plots than are shown in the final slides.
Additional figures serve as diagnostics, robustness checks, and appendix material.
Main-paper figures are selected from this full set.
"""

# Helpers

def savefig(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)

def _clean_kclip(x):
    if pd.isna(x):
        return "None"
    s = str(x).strip().lower()
    if s in ["none", "nan", ""]:
        return "None"
    try:
        v = float(s)
        if np.isfinite(v):
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return str(v)
    except Exception:
        pass
    return str(x).strip()

def _clean_temp(x):
    if pd.isna(x):
        return 1.0
    s = str(x).strip().lower()
    if s in ["none", "nan", ""]:
        return 1.0
    try:
        return float(s)
    except Exception:
        return 1.0

def mean_over_seeds(df, group_cols, value_cols):
    return df.groupby(group_cols + ["epoch"], as_index=False)[value_cols].mean()

def plot_seed_lines_plus_mean(
    df, group_key, x="epoch", y="ELBO", title="", xlabel="epoch", ylabel=None,
    legend_title=None, logy=False, fname="plot.png"
):
    fig, ax = plt.subplots()
    if logy:
        ax.set_yscale("log")

    # thin seed lines to check if effect is consistent
    for gval, g in df.groupby(group_key):
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values(x)
            ax.plot(gs[x], gs[y], alpha=0.35, linewidth=1)

    # thick mean lines 
    mean_df = df.groupby([group_key, x], as_index=False)[y].mean()
    for gval, gm in mean_df.groupby(group_key):
        gm = gm.sort_values(x)
        ax.plot(gm[x], gm[y], linewidth=2.5, label=f"{group_key}={gval}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or y)
    ax.legend(title=legend_title) if legend_title else ax.legend()
    savefig(fname)

def final_epoch_summary_bar(df, group_col, metrics, title, fname, sort_by=None):
    last_ep = df["epoch"].max()
    d = df[df["epoch"] == last_ep].copy()

    agg = d.groupby([group_col, "seed"], as_index=False)[metrics].mean()
    mean = agg.groupby(group_col, as_index=False)[metrics].mean()
    std  = agg.groupby(group_col, as_index=False)[metrics].std(ddof=0)

    if sort_by is not None and sort_by in mean.columns:
        order = mean.sort_values(sort_by)[group_col].tolist()
        mean[group_col] = pd.Categorical(mean[group_col], categories=order, ordered=True)
        std[group_col]  = pd.Categorical(std[group_col], categories=order, ordered=True)
        mean = mean.sort_values(group_col)
        std  = std.sort_values(group_col)

    for m in metrics:
        fig, ax = plt.subplots()
        xlabels = mean[group_col].astype(str).tolist()
        x = np.arange(len(xlabels))
        ax.bar(x, mean[m].values, yerr=std[m].values, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=25, ha="right")
        ax.set_title(f"{title}\n(final epoch: {last_ep}) — {m}")
        ax.set_ylabel(m)
        savefig(fname.replace(".png", f"_{m}.png"))

# stress overlay plot (baseline + temp + clip on the same plot)
def plot_stress_overlay(stress_df, y, title, fname, logy=False, show_seed_lines=True):
    fig, ax = plt.subplots()
    if logy:
        ax.set_yscale("log")

    # optional thin seed lines
    if show_seed_lines:
        for setting, g in stress_df.groupby("setting"):
            for seed, gs in g.groupby("seed"):
                gs = gs.sort_values("epoch")
                ax.plot(gs["epoch"], gs[y], alpha=0.20, linewidth=1)

    # thick mean lines (mean over seeds)
    mean_df = stress_df.groupby(["setting", "epoch"], as_index=False)[y].mean()
    for setting, gm in mean_df.groupby("setting"):
        gm = gm.sort_values("epoch")
        ax.plot(gm["epoch"], gm[y], linewidth=2.5, label=setting)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(y)
    ax.legend()
    savefig(fname)

# stress scatter (mechanism link) y vs x with mean point per epoch (or all points)
def plot_scatter(stress_df, x, y, title, fname, logx=False, logy=False):
    fig, ax = plt.subplots()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.scatter(stress_df[x], stress_df[y], alpha=0.35, s=18)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    savefig(fname)

# Load + preprocess
df = pd.read_csv(CSV_PATH)

# Guard against missing columns
for col in ["kappa_clip", "kappa_temp"]:
    if col not in df.columns:
        df[col] = np.nan

df["kappa_clip"] = df["kappa_clip"].apply(_clean_kclip)
df["kappa_temp"] = df["kappa_temp"].apply(_clean_temp)

print("Unique kappa_clip:", sorted(df["kappa_clip"].dropna().unique()))
print("Unique kappa_temp:", sorted(pd.Series(df["kappa_temp"].dropna().unique()).tolist()))

# enforce dtypes
df["model_type"] = df["model_type"].astype(str)
df["z_dim"] = df["z_dim"].astype(int)
df["z_dim_effective"] = df["z_dim_effective"].astype(int)
df["seed"] = df["seed"].astype(int)
df["epoch"] = df["epoch"].astype(int)

df_normal = df[df["model_type"] == "normal"].copy()
df_vmf    = df[df["model_type"] == "vmf"].copy()

# baseline = no clip + T=1.0
vmf_base = df_vmf[(df_vmf["kappa_clip"] == "None") & (df_vmf["kappa_temp"] == 1.0)].copy()


# ELBO: Normal vs vMF baseline

value_cols_common = ["ELBO", "LL", "train_seconds_per_batch", "Delta_sigmoid", "KL_per_dim"]
value_cols_vmf = value_cols_common + [
    "accept_rate_mean",
    "attempts_p90", "attempts_max",
    "kappa_obs_mean", "kappa_obs_p90", "kappa_obs_max",
]

normal_mean = mean_over_seeds(df_normal, ["z_dim"], value_cols_common)
vmf_mean    = mean_over_seeds(vmf_base,  ["z_dim"], value_cols_vmf)

fig, ax = plt.subplots()
for z, g in normal_mean.groupby("z_dim"):
    g = g.sort_values("epoch")
    ax.plot(g["epoch"], g["ELBO"], label=f"Normal z={z}")
for z, g in vmf_mean.groupby("z_dim"):
    g = g.sort_values("epoch")
    ax.plot(g["epoch"], g["ELBO"], label=f"vMF z={z} (eff={z+1})")
ax.set_title("ELBO vs epoch (Normal vs vMF baseline)")
ax.set_xlabel("epoch")
ax.set_ylabel("ELBO")
ax.legend()
savefig("01_ELBO_normal_vs_vmf.png")


# LL: Normal vs vMF baseline

fig, ax = plt.subplots()
for z, g in normal_mean.groupby("z_dim"):
    g = g.sort_values("epoch")
    ax.plot(g["epoch"], g["LL"], label=f"Normal z={z}")
for z, g in vmf_mean.groupby("z_dim"):
    g = g.sort_values("epoch")
    ax.plot(g["epoch"], g["LL"], label=f"vMF z={z} (eff={z+1})")
ax.set_title("MC log-likelihood vs epoch (Normal vs vMF baseline)")
ax.set_xlabel("epoch")
ax.set_ylabel("LL (MC estimate)")
ax.legend()
savefig("02_LL_normal_vs_vmf.png")


# κ dynamics (baseline)

plot_seed_lines_plus_mean(
    vmf_base, group_key="z_dim", y="kappa_obs_mean",
    title="κ mean over epochs (vMF baseline) — seed lines + mean",
    ylabel="κ (mean)", logy=True,
    fname="03a_kappa_mean_seed_plus_mean_log.png",
)

plot_seed_lines_plus_mean(
    vmf_base, group_key="z_dim", y="kappa_obs_max",
    title="κ max over epochs (vMF baseline) — seed lines + mean (tail!)",
    ylabel="κ (max)", logy=True,
    fname="03b_kappa_max_seed_plus_mean_log.png",
)


# Attempts tail (baseline)

plot_seed_lines_plus_mean(
    vmf_base, group_key="z_dim", y="attempts_p90",
    title="Sampling attempts p90 over epochs (vMF baseline)",
    ylabel="attempts_p90", logy=True,
    fname="04a_attempts_p90_seed_plus_mean_log.png",
)

plot_seed_lines_plus_mean(
    vmf_base, group_key="z_dim", y="attempts_max",
    title="Sampling attempts MAX over epochs (vMF baseline) — tail behavior",
    ylabel="attempts_max", logy=True,
    fname="04b_attempts_max_seed_plus_mean_log.png",
)


# Time per batch : Normal vs vMF baseline

fig, ax = plt.subplots()
for z, g in normal_mean.groupby("z_dim"):
    g = g.sort_values("epoch")
    ax.plot(g["epoch"], g["train_seconds_per_batch"], label=f"Normal z={z}")
for z, g in vmf_mean.groupby("z_dim"):
    g = g.sort_values("epoch")
    ax.plot(g["epoch"], g["train_seconds_per_batch"], label=f"vMF z={z}")
ax.set_title("Training cost: seconds per batch (Normal vs vMF baseline)")
ax.set_xlabel("epoch")
ax.set_ylabel("sec/batch")
ax.legend()
savefig("05_time_per_batch.png")


# Delta_sigmoid : Normal vs vMF baseline

fig, ax = plt.subplots()
for z, g in normal_mean.groupby("z_dim"):
    g = g.sort_values("epoch")
    ax.plot(g["epoch"], g["Delta_sigmoid"], label=f"Normal z={z}")
for z, g in vmf_mean.groupby("z_dim"):
    g = g.sort_values("epoch")
    ax.plot(g["epoch"], g["Delta_sigmoid"], label=f"vMF z={z} (eff={z+1})")
ax.set_title("Δ (posterior resampling sensitivity) vs epoch — Normal vs vMF baseline")
ax.set_xlabel("epoch")
ax.set_ylabel("Delta_sigmoid")
ax.legend()
savefig("06_delta_sigmoid_normal_vs_vmf.png")




fig, ax = plt.subplots()
g = vmf_mean[vmf_mean["z_dim"] == 20].sort_values("epoch")
ax.plot(g["epoch"], g["Delta_sigmoid"], label="vMF z=20", linewidth=2.5)
ax.set_title("Δ sigmoid over epoch — vMF z=20")
ax.set_xlabel("epoch")
ax.set_ylabel("Δ sigmoid")
ax.legend()
savefig("delta_zoom_vmf_z20.png")



# Stress setting (z=20): temperature + clipping (same plot)

stress = df_vmf[df_vmf["z_dim"] == 20].copy()

def setting_label(r):
    clip = r["kappa_clip"]
    T = float(r["kappa_temp"])
    if clip != "None":
        return f"clip={clip},T=1.0"
    return f"clip=None,T={T:.1f}"

stress["setting"] = stress.apply(setting_label, axis=1)

keep_settings = [
    "clip=None,T=1.0",
    "clip=50,T=1.0",
    "clip=100,T=1.0",
    "clip=None,T=0.5",
    "clip=None,T=2.0",
]
stress = stress[stress["setting"].isin(keep_settings)].copy()

print("Stress settings present:", sorted(stress["setting"].unique()))
print("Rows per setting:\n", stress["setting"].value_counts())

# same plot: (mean-over-seeds) 
plot_stress_overlay(
    stress, y="kappa_obs_mean",
    title="Stress z=20: κ mean — baseline vs temperature vs clipping",
    fname="S1_stress_overlay_kappa_mean_log.png",
    logy=True,
    show_seed_lines=True
)

plot_stress_overlay(
    stress, y="attempts_p90",
    title="Stress z=20: attempts p90 — baseline vs temperature vs clipping",
    fname="S2_stress_overlay_attempts_p90_log.png",
    logy=True,
    show_seed_lines=True
)

plot_stress_overlay(
    stress, y="train_seconds_per_batch",
    title="Stress z=20: seconds/batch — baseline vs temperature vs clipping",
    fname="S3_stress_overlay_seconds_per_batch.png",
    logy=False,
    show_seed_lines=True
)

plot_stress_overlay(
    stress, y="Delta_sigmoid",
    title="Stress z=20: Δsigmoid — baseline vs temperature vs clipping",
    fname="S4_stress_overlay_delta_sigmoid.png",
    logy=False,
    show_seed_lines=True
)

# Optional: we could also show κ p90 (stronger than mean)
if "kappa_obs_p90" in stress.columns:
    plot_stress_overlay(
        stress, y="kappa_obs_p90",
        title="Stress z=20: κ p90 — baseline vs temperature vs clipping",
        fname="S1b_stress_overlay_kappa_p90_log.png",
        logy=True,
        show_seed_lines=True
    )

# Final-epoch summaries (keep for appendix)

final_epoch_summary_bar(
    stress,
    group_col="setting",
    metrics=["attempts_max", "train_seconds_per_batch"],
    title="Stress (z=20) — κ-clipping vs temperature (sampler tail + cost)",
    fname="A1_stress_finalepoch_sampler_cost.png",
    sort_by="attempts_max",
)

final_epoch_summary_bar(
    stress,
    group_col="setting",
    metrics=["Delta_sigmoid", "LL"],
    title="Stress (z=20) — κ-clipping vs temperature (Δ + LL)",
    fname="A2_stress_finalepoch_delta_ll.png",
    sort_by="Delta_sigmoid",
)


# attempts_p90 vs κ_mean (mechanism link)
plot_scatter(
    stress, x="kappa_obs_mean", y="attempts_p90",
    title="Stress z=20: attempts p90 vs κ mean (all epochs, all seeds)",
    fname="X1_scatter_attempts_p90_vs_kappa_mean.png",
    logx=True, logy=True
)

# time vs attempts_p90 (compute consequence)
plot_scatter(
    stress, x="attempts_p90", y="train_seconds_per_batch",
    title="Stress z=20: seconds/batch vs attempts p90 (all epochs, all seeds)",
    fname="X2_scatter_time_vs_attempts_p90.png",
    logx=True, logy=False
)

# delta vs attempts_p90 (latent usage vs sampling pain)
plot_scatter(
    stress, x="attempts_p90", y="Delta_sigmoid",
    title="Stress z=20: Δsigmoid vs attempts p90 (all epochs, all seeds)",
    fname="X3_scatter_delta_vs_attempts_p90.png",
    logx=True, logy=False
)

print("\nAll plots saved to:", OUT_DIR)


# Appendix: MNIST vs FashionMNIST (just as robustness check)

def make_setting(df):
    def label_row(r):
        clip = r["kappa_clip"]
        T = float(r["kappa_temp"])
        if clip != "None":
            return f"clip={clip},T=1.0"
        return f"clip=None,T={T:.1f}"
    df = df.copy()
    df["setting"] = df.apply(label_row, axis=1)
    return df

keep_settings = {
    "clip=None,T=1.0",
    "clip=None,T=0.5",
    "clip=None,T=2.0",
    "clip=50,T=1.0",
    "clip=100,T=1.0",
}

def plot_kappa_mnist_vs_fmnist(df_vmf, z=20, metric="kappa_obs_mean", fname="APPX_kappa_mnist_vs_fmnist.png"):
    # filter to stress setting
    d = df_vmf[df_vmf["z_dim"] == z].copy()

    # require dataset column
    if "dataset" not in d.columns:
        raise ValueError("CSV has no 'dataset' column. Add it during training or save MNIST/FMNIST to separate CSVs.")

    d["dataset"] = d["dataset"].astype(str).str.lower()
    d = make_setting(d)
    d = d[d["setting"].isin(keep_settings)].copy()

    # mean over seeds for clean appendix plot
    mean_df = d.groupby(["dataset", "setting", "epoch"], as_index=False)[metric].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ds in zip(axes, ["mnist", "fashionmnist"]):
        sub = mean_df[mean_df["dataset"] == ds].copy()
        for setting, g in sub.groupby("setting"):
            g = g.sort_values("epoch")
            ax.plot(g["epoch"], g[metric], linewidth=2, label=setting)

        ax.set_title(ds.upper())
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(metric)
    axes[1].legend(loc="best", fontsize=8, title="Setting")
    plt.suptitle(f"Appendix: κ dynamics on MNIST vs FashionMNIST (vMF, z={z})")
    savefig(fname)

# call it
plot_kappa_mnist_vs_fmnist(df_vmf, z=20, metric="kappa_obs_mean", fname="APPX_kappa_mnist_vs_fmnist.png")


# Standalone plot: Sampling attempts MEAN (vMF baseline)

if "attempts_mean" not in vmf_base.columns:
    raise ValueError("attempts_mean not found in CSV — cannot plot mean attempts.")

fig, ax = plt.subplots()

# thin seed lines
for z, g in vmf_base.groupby("z_dim"):
    for seed, gs in g.groupby("seed"):
        gs = gs.sort_values("epoch")
        ax.plot(
            gs["epoch"],
            gs["attempts_mean"],
            alpha=0.30,
            linewidth=1
        )

# thick mean-over-seeds line
mean_df = (
    vmf_base
    .groupby(["z_dim", "epoch"], as_index=False)["attempts_mean"]
    .mean()
)

for z, gm in mean_df.groupby("z_dim"):
    gm = gm.sort_values("epoch")
    ax.plot(
        gm["epoch"],
        gm["attempts_mean"],
        linewidth=2.5,
        label=f"z={z}"
    )

ax.set_title("Sampling attempts (MEAN) over epochs — vMF baseline")
ax.set_xlabel("epoch")
ax.set_ylabel("attempts_mean")
ax.set_yscale("log")
ax.legend()

savefig("04c_attempts_mean_ONLY_seed_plus_mean_log.png")
