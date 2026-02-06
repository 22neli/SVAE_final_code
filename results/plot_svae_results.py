# plot_svae_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter
import matplotlib.cm as cm

# ============================================================
# Paths / config
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "results_svae_sweep.csv"

OUT_DIR    = BASE_DIR / "plots_svae"
OUT_MAIN   = OUT_DIR / "main"
OUT_STRESS = OUT_DIR / "stress"
OUT_DIAG   = OUT_DIR / "diagnostics"
OUT_APPX   = OUT_DIR / "appendix"
for d in [OUT_DIR, OUT_MAIN, OUT_STRESS, OUT_DIAG, OUT_APPX]:
    d.mkdir(exist_ok=True)

MAIN_DATASET = "mnist"      # set to "fashionmnist" if we want ALL main plots on FashionMNIST
Z_STRESS     = 20


BATCH_SIZE = 64

"""
NOTE:
We intentionally generate more plots than are shown in the final slides.
Additional figures serve as diagnostics, robustness checks, and appendix material.
Main-paper figures are selected from this full set.
"""


# Publication-style tuning 

# Mean curves (main lines)
LW_MEAN = 1.7
MS_MEAN = 3.8
# Prettier dashed lines for vMF (longer, calmer than "--")
VMF_DASH = (0, (10, 4))   

MARKEVERY_MEAN = 1 

# Seed curves (thin, very light)
LW_SEED = 0.8
MS_SEED = 2.0
MARKEVERY_SEED = 3
ALPHA_SEED = 0.13
plt.rcParams["lines.dash_capstyle"] = "round"
plt.rcParams["lines.solid_capstyle"] = "round"
plt.rcParams["lines.dash_joinstyle"] = "round"
plt.rcParams["lines.solid_joinstyle"] = "round"

# Labels (no underscores; explain averages in title, not "(mean)")

YLABEL = {
    "ELBO": "ELBO",
    "LL": "Log-likelihood (MC estimate)",
    "train_seconds_per_batch": "Seconds per batch",
    "runtime_ms": "Milliseconds per batch",
    "Delta_sigmoid": "Δ (sigmoid output)",
    "KL_per_dim": "KL per latent dimension",
    "accept_rate_mean": "Acceptance rate",
    "attempts_mean": "Attempts (mean)",
    "attempts_p90": "Attempts (p90)",
    "attempts_max": "Attempts (max)",
    "kappa_obs_mean": "κ",
    "kappa_obs_p90": "κ (p90)",
    "kappa_obs_max": "κ (max)",
}


# Colors: keep “light -> dark with increasing z”, but avoid too-pale hues

def _palette_from_cmap(cmap_name: str, n: int, lo: float = 0.45, hi: float = 0.92):
    """
    Sample n colors from a colormap (lo->hi).
    lo>=0.45 avoids near-white tones that wash out on slides/projectors.
    """
    cmap = cm.get_cmap(cmap_name)
    xs = np.linspace(lo, hi, n)
    return [cmap(x) for x in xs]

# z dims assumed {5,10,20} -> 3 shades
NORMAL_COLORS = _palette_from_cmap("Blues",   3, lo=0.48, hi=0.92)
VMF_COLORS    = _palette_from_cmap("Oranges", 3, lo=0.48, hi=0.95)


# Cleaning helpers

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


# Plot style helpers 

def _format_epoch_axis(ax, max_epoch: int):
    # Discrete epochs: majors every 5, minors every 1 
    ax.set_xlim(1, max_epoch)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", axis="x", alpha=0.22)
    ax.grid(True, which="minor", axis="x", alpha=0.10)

def _format_y_grid(ax, logy: bool):
    ax.grid(True, which="major", axis="y", alpha=0.18)
    if logy:
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(False, which="minor", axis="y")

def _apply_style(ax, max_epoch: int, logy: bool):
    if logy:
        ax.set_yscale("log")
    _format_epoch_axis(ax, max_epoch)
    _format_y_grid(ax, logy)

def _savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()
    print("Saved:", path)

def mean_over_seeds(df, group_cols, value_cols):
    return df.groupby(group_cols + ["epoch"], as_index=False)[value_cols].mean()


# Consistent encoding: Normal vs vMF 

# Normal = solid + circles, vMF = dashed + triangles
MODEL_STYLE = {
    "normal": {"marker": "o"},
    "vmf":    {"marker": "^"},
}

def _z_color_map(unique_z_sorted, model_type: str):
    pal = NORMAL_COLORS if model_type == "normal" else VMF_COLORS
    return {z: pal[i % len(pal)] for i, z in enumerate(unique_z_sorted)}

# Stress overlays: keep vMF marker "^" always, encode intervention type with linestyle
# Pretty dash patterns for stress/appendix settings
BASE_LINE = "-"                 # baseline: solid
CLIP_DASH = (0, (10, 4, 2, 4))  # clip: dash–dot
TEMP_DASH = (0, (1, 3))         # temp: dotted


def _stress_linestyle(setting: str):
    # baseline vs clip vs temperature 
    if "clip =" in setting and "clip = None" not in setting:
        return CLIP_DASH  # clipping
    if "T =" in setting and "T = 1.0" not in setting:
        return TEMP_DASH    # temperature scaling
    return BASE_LINE      # baseline

def _plot_mean_curves(ax, mean_df, group_key, y, label_fn, model_type=None):
    """
    mean_df must have columns: epoch, group_key, y
    model_type selects marker; linestyle set here by model type (solid vs dashed).
    """
    if len(mean_df) == 0:
        return

    marker = MODEL_STYLE.get(model_type, {"marker": "o"})["marker"]
    ls = "-" if model_type == "normal" else VMF_DASH
    groups_sorted = sorted(mean_df[group_key].unique().tolist())
    color_map = _z_color_map(groups_sorted, model_type=model_type)

    for gval, g in mean_df.groupby(group_key):
        g = g.sort_values("epoch")
        ax.plot(
            g["epoch"], g[y],
            linestyle=ls,
            marker=marker,
            markersize=MS_MEAN,
            markevery=MARKEVERY_MEAN,
            linewidth=LW_MEAN,
            color=color_map.get(gval, None),
            label=label_fn(gval),
        )

def _plot_seed_lines_plus_mean(
    df, group_key, y, title, out_path, logy=False, ylabel=None, mean_model_type=None
):
    if len(df) == 0 or y not in df.columns:
        return

    max_epoch = int(df["epoch"].max())
    fig, ax = plt.subplots()

    marker = MODEL_STYLE.get(mean_model_type, {"marker": "o"})["marker"]
    ls = "-" if mean_model_type == "normal" else VMF_DASH
    groups_sorted = sorted(df[group_key].unique().tolist())
    color_map = _z_color_map(groups_sorted, model_type=mean_model_type)

    # thin seed lines
    for gval, g in df.groupby(group_key):
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values("epoch")
            ax.plot(
                gs["epoch"], gs[y],
                linestyle=ls,
                marker=marker,
                markersize=MS_SEED,
                markevery=MARKEVERY_SEED,
                linewidth=LW_SEED,
                alpha=ALPHA_SEED,
                color=color_map.get(gval, None),
            )

    # mean line
    mean_df = df.groupby([group_key, "epoch"], as_index=False)[y].mean()
    _plot_mean_curves(
        ax, mean_df, group_key, y,
        label_fn=lambda gval: f"{group_key} = {gval}",
        model_type=mean_model_type,
    )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or YLABEL.get(y, y.replace("_", " ")))
    _apply_style(ax, max_epoch=max_epoch, logy=logy)
    ax.legend()
    _savefig(out_path)

def _plot_stress_overlay(
    stress_df, y, title, out_path, logy=False, show_seed_lines=True, ylabel=None
):
    if len(stress_df) == 0 or y not in stress_df.columns:
        return

    max_epoch = int(stress_df["epoch"].max())
    fig, ax = plt.subplots()

    marker = MODEL_STYLE["vmf"]["marker"]

    # seed lines (light)
    if show_seed_lines:
        for setting, g in stress_df.groupby("setting"):
            ls = _stress_linestyle(setting)
            for seed, gs in g.groupby("seed"):
                gs = gs.sort_values("epoch")
                ax.plot(
                    gs["epoch"], gs[y],
                    linestyle=ls,
                    marker=marker,
                    markersize=MS_SEED,
                    markevery=MARKEVERY_SEED,
                    linewidth=LW_SEED,
                    alpha=ALPHA_SEED,
                )

    # mean curves
    mean_df = stress_df.groupby(["setting", "epoch"], as_index=False)[y].mean()
    settings_sorted = sorted(mean_df["setting"].unique())
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(settings_sorted)}

    for setting, gm in mean_df.groupby("setting"):
        gm = gm.sort_values("epoch")
        ls = _stress_linestyle(setting)
        ax.plot(
            gm["epoch"], gm[y],
            linestyle=ls,
            marker=marker,
            markersize=MS_MEAN,
            markevery=MARKEVERY_MEAN,
            linewidth=max(LW_MEAN, 1.8),
            color=color_map[setting],
            label=setting,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or YLABEL.get(y, y.replace("_", " ")))
    _apply_style(ax, max_epoch=max_epoch, logy=logy)

    # Legend ordering: baseline first, then clipping, then temperature
    handles, labels = ax.get_legend_handles_labels()

    def _legend_key(label):
        if "clip = None, T = 1.0" in label:
            return 0
        if "clip =" in label:
            return 1
        if "T =" in label:
            return 2
        return 3

    order = sorted(range(len(labels)), key=lambda i: _legend_key(labels[i]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=9)

    _savefig(out_path)

def _plot_scatter(df, x, y, title, out_path, logx=False, logy=False, xlabel=None, ylabel=None):
    # markers only; do NOT connect points.
    if len(df) == 0 or x not in df.columns or y not in df.columns:
        return
    fig, ax = plt.subplots()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.scatter(df[x], df[y], alpha=0.40, s=24, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel or x.replace("_", " "))
    ax.set_ylabel(ylabel or y.replace("_", " "))
    ax.grid(True, which="major", alpha=0.18)
    _savefig(out_path)

def _final_epoch_summary_bar(df, group_col, metrics, title, out_prefix, sort_by=None):
    if len(df) == 0:
        return

    last_ep = int(df["epoch"].max())
    d = df[df["epoch"] == last_ep].copy()
    if len(d) == 0:
        return

    keep_metrics = [m for m in metrics if m in d.columns]
    if not keep_metrics:
        return

    per_seed = d.groupby([group_col, "seed"], as_index=False)[keep_metrics].mean()
    mean = per_seed.groupby(group_col, as_index=False)[keep_metrics].mean()
    std  = per_seed.groupby(group_col, as_index=False)[keep_metrics].std(ddof=0)

    if sort_by is not None and sort_by in mean.columns:
        order = mean.sort_values(sort_by)[group_col].tolist()
        mean[group_col] = pd.Categorical(mean[group_col], categories=order, ordered=True)
        std[group_col]  = pd.Categorical(std[group_col], categories=order, ordered=True)
        mean = mean.sort_values(group_col)
        std  = std.sort_values(group_col)

    for m in keep_metrics:
        fig, ax = plt.subplots()
        xlabels = mean[group_col].astype(str).tolist()
        x = np.arange(len(xlabels))
        ax.bar(x, mean[m].values, yerr=std[m].values, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=25, ha="right")
        ax.set_title(f"{title}\nFinal epoch = {last_ep} (mean ± std over seeds)")

        ax.set_ylabel(YLABEL.get(m, m.replace("_", " ")))
        ax.grid(True, which="major", axis="y", alpha=0.18)
        _savefig(out_prefix.with_name(out_prefix.name + f"_{m}.png"))


# Load + preprocess

df = pd.read_csv(CSV_PATH)

# ensure columns exist
for col in ["kappa_clip", "kappa_temp", "dataset", "model_type"]:
    if col not in df.columns:
        df[col] = np.nan

df["kappa_clip"] = df["kappa_clip"].apply(_clean_kclip)
df["kappa_temp"] = df["kappa_temp"].apply(_clean_temp)

# dtypes (avoid sorting/grouping weirdness)
df["model_type"] = df["model_type"].astype(str).str.lower()
df["dataset"] = df["dataset"].astype(str).str.lower()
for col in ["z_dim", "z_dim_effective", "seed", "epoch"]:
    if col in df.columns:
        df[col] = df[col].astype(int)

print("Datasets:", sorted(df["dataset"].unique()))
print("kappa_clip:", sorted(df["kappa_clip"].dropna().unique()))
print("kappa_temp:", sorted(pd.Series(df["kappa_temp"].dropna().unique()).tolist()))


# Split: all vs main dataset

df_normal_all = df[df["model_type"] == "normal"].copy()
df_vmf_all    = df[df["model_type"] == "vmf"].copy()

df_main   = df[df["dataset"] == MAIN_DATASET].copy()
df_normal = df_main[df_main["model_type"] == "normal"].copy()
df_vmf    = df_main[df_main["model_type"] == "vmf"].copy()

# baseline vMF: no clip + T=1.0 (within MAIN dataset only)
vmf_base = df_vmf[(df_vmf["kappa_clip"] == "None") & (df_vmf["kappa_temp"] == 1.0)].copy()

seeds_present = sorted(df_main["seed"].unique().tolist()) if len(df_main) and "seed" in df_main.columns else []
seed_note = f"Averaged over seeds {seeds_present}" if seeds_present else "Averaged over seeds"

# Metric sets (only keep columns that exist)
value_cols_common = ["ELBO", "LL", "train_seconds_per_batch", "Delta_sigmoid", "KL_per_dim"]
value_cols_vmf = value_cols_common + [
    "accept_rate_mean",
    "attempts_mean", "attempts_p90", "attempts_max",
    "kappa_obs_mean", "kappa_obs_p90", "kappa_obs_max",
]

common_cols_present = [c for c in value_cols_common if c in df.columns]
vmf_cols_present    = [c for c in value_cols_vmf if c in df.columns]

# MAIN plots: Normal vs vMF baseline (mean over seeds)
normal_mean = mean_over_seeds(
    df_normal, ["z_dim"], [c for c in common_cols_present if c in df_normal.columns]
) if len(df_normal) else pd.DataFrame()

vmf_mean = mean_over_seeds(
    vmf_base, ["z_dim"], [c for c in vmf_cols_present if c in vmf_base.columns]
) if len(vmf_base) else pd.DataFrame()

max_epoch_main = int(df_main["epoch"].max()) if len(df_main) else 1


# 0) ELBO + LL together (shared y-axis) — like appendix style
if (
    len(normal_mean) and len(vmf_mean)
    and "ELBO" in normal_mean.columns and "ELBO" in vmf_mean.columns
    and "LL"   in normal_mean.columns and "LL"   in vmf_mean.columns
):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4), sharey=True, constrained_layout=True
    )

    # --- Left: ELBO ---
    _plot_mean_curves(
        ax1, normal_mean, "z_dim", "ELBO",
        label_fn=lambda z: f"Normal  z = {z}",
        model_type="normal",
    )
    _plot_mean_curves(
        ax1, vmf_mean, "z_dim", "ELBO",
        label_fn=lambda z: f"vMF  z = {z} (eff = {z+1})",
        model_type="vmf",
    )
    ax1.set_title("a) ELBO")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metric value")
    _apply_style(ax1, max_epoch=max_epoch_main, logy=False)

    # --- Right: LL ---
    _plot_mean_curves(
        ax2, normal_mean, "z_dim", "LL",
        label_fn=lambda z: f"Normal  z = {z}",
        model_type="normal",
    )
    _plot_mean_curves(
        ax2, vmf_mean, "z_dim", "LL",
        label_fn=lambda z: f"vMF  z = {z} (eff = {z+1})",
        model_type="vmf",
    )
    ax2.set_title("b) MC log-likelihood")
    ax2.set_xlabel("Epoch")
    _apply_style(ax2, max_epoch=max_epoch_main, logy=False)

    # One shared legend 
    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(
        handles, labels,
        loc="lower right",
        fontsize=9,
        frameon=True
    )


    fig.suptitle(f"ELBO and MC log-likelihood over training — {MAIN_DATASET.upper()}\n{seed_note}")

    out_path = OUT_MAIN / "00_ELBO_LL_shared_yaxis.png"
    plt.savefig(out_path, dpi=240)
    plt.close()
    print("Saved:", out_path)


# 1) ELBO
if len(normal_mean) and "ELBO" in normal_mean.columns and len(vmf_mean) and "ELBO" in vmf_mean.columns:
    fig, ax = plt.subplots()
    _plot_mean_curves(
        ax, normal_mean, "z_dim", "ELBO",
        label_fn=lambda z: f"Normal  z = {z}",
        model_type="normal",
    )
    _plot_mean_curves(
        ax, vmf_mean, "z_dim", "ELBO",
        label_fn=lambda z: f"vMF  z = {z} (eff = {z+1})",
        model_type="vmf",
    )
    ax.set_title(f"ELBO over training — {MAIN_DATASET.upper()}\n{seed_note}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(YLABEL["ELBO"])
    _apply_style(ax, max_epoch=max_epoch_main, logy=False)
    ax.legend()
    _savefig(OUT_MAIN / "01_ELBO_normal_vs_vmf.png")

# 2) LL
if len(normal_mean) and "LL" in normal_mean.columns and len(vmf_mean) and "LL" in vmf_mean.columns:
    fig, ax = plt.subplots()
    _plot_mean_curves(ax, normal_mean, "z_dim", "LL",
                      label_fn=lambda z: f"Normal  z = {z}", model_type="normal")
    _plot_mean_curves(ax, vmf_mean, "z_dim", "LL",
                      label_fn=lambda z: f"vMF  z = {z} (eff = {z+1})", model_type="vmf")
    ax.set_title(f"Monte Carlo log-likelihood over training — {MAIN_DATASET.upper()}\n{seed_note}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(YLABEL["LL"])
    _apply_style(ax, max_epoch=max_epoch_main, logy=False)
    ax.legend()
    _savefig(OUT_MAIN / "02_LL_normal_vs_vmf.png")

# 3) Runtime per batch (ms)
if (
    len(normal_mean) and len(vmf_mean)
    and "train_seconds_per_batch" in normal_mean.columns
    and "train_seconds_per_batch" in vmf_mean.columns
):
    normal_ms = normal_mean.copy()
    vmf_ms = vmf_mean.copy()
    normal_ms["runtime_ms"] = 1000.0 * normal_ms["train_seconds_per_batch"]
    vmf_ms["runtime_ms"]    = 1000.0 * vmf_ms["train_seconds_per_batch"]

    fig, ax = plt.subplots()
    _plot_mean_curves(ax, normal_ms, "z_dim", "runtime_ms",
                      label_fn=lambda z: f"Normal  z = {z}", model_type="normal")
    _plot_mean_curves(ax, vmf_ms, "z_dim", "runtime_ms",
                      label_fn=lambda z: f"vMF  z = {z}", model_type="vmf")

    ax.set_title(
        f"Runtime per batch (ms) over training — {MAIN_DATASET.upper()}\n"
        f"{seed_note}; averaged over all training mini-batches per epoch; batch size = {BATCH_SIZE}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(YLABEL["runtime_ms"])
    _apply_style(ax, max_epoch=max_epoch_main, logy=False)
    ax.legend()
    _savefig(OUT_MAIN / "03_runtime_ms_per_batch.png")

# 4) Delta
if (
    len(normal_mean) and len(vmf_mean)
    and "Delta_sigmoid" in normal_mean.columns
    and "Delta_sigmoid" in vmf_mean.columns
):
    fig, ax = plt.subplots()
    _plot_mean_curves(ax, normal_mean, "z_dim", "Delta_sigmoid",
                      label_fn=lambda z: f"Normal  z = {z}", model_type="normal")
    _plot_mean_curves(ax, vmf_mean, "z_dim", "Delta_sigmoid",
                      label_fn=lambda z: f"vMF  z = {z} (eff = {z+1})", model_type="vmf")
    ax.set_title(f"Latent usage diagnostic Δ over training — {MAIN_DATASET.upper()}\n{seed_note}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(YLABEL["Delta_sigmoid"])
    _apply_style(ax, max_epoch=max_epoch_main, logy=False)
    ax.legend()
    _savefig(OUT_MAIN / "04_delta_over_training.png")

# Baseline vMF dynamics: seed lines + mean (with markers)
# 
if len(vmf_base):
    _plot_seed_lines_plus_mean(
        vmf_base, group_key="z_dim", y="kappa_obs_mean",
        title=f"κ over training — vMF baseline, {MAIN_DATASET.upper()}\nThin: seeds; thick: mean",
        out_path=OUT_MAIN / "05_kappa_mean_seed_plus_mean.png",
        logy=True, ylabel=YLABEL["kappa_obs_mean"], mean_model_type="vmf",
    )
    _plot_seed_lines_plus_mean(
        vmf_base, group_key="z_dim", y="kappa_obs_max",
        title=f"κ (max) over training — vMF baseline, {MAIN_DATASET.upper()}\nThin: seeds; thick: mean",
        out_path=OUT_MAIN / "06_kappa_max_seed_plus_mean.png",
        logy=True, ylabel=YLABEL["kappa_obs_max"], mean_model_type="vmf",
    )
    _plot_seed_lines_plus_mean(
        vmf_base, group_key="z_dim", y="attempts_p90",
        title=f"Sampling attempts (p90) over training — vMF baseline, {MAIN_DATASET.upper()}\nThin: seeds; thick: mean",
        out_path=OUT_MAIN / "07_attempts_p90_seed_plus_mean.png",
        logy=True, ylabel=YLABEL["attempts_p90"], mean_model_type="vmf",
    )
    _plot_seed_lines_plus_mean(
        vmf_base, group_key="z_dim", y="attempts_max",
        title=f"Sampling attempts (max) over training — vMF baseline, {MAIN_DATASET.upper()}\nThin: seeds; thick: mean",
        out_path=OUT_MAIN / "08_attempts_max_seed_plus_mean.png",
        logy=True, ylabel=YLABEL["attempts_max"], mean_model_type="vmf",
    )


# Stress setting (z = Z_STRESS): overlay baseline vs clip vs temperature

stress = df_vmf[df_vmf["z_dim"] == Z_STRESS].copy()

def _setting_label(row):
    clip = row["kappa_clip"]
    T = float(row["kappa_temp"])
    if clip != "None":
        return f"clip = {clip}, T = 1.0"
    return f"clip = None, T = {T:.1f}"

if len(stress):
    stress["setting"] = stress.apply(_setting_label, axis=1)

    keep_settings = [
        "clip = None, T = 1.0",
        "clip = 50, T = 1.0",
        "clip = 100, T = 1.0",
        "clip = None, T = 0.5",
        "clip = None, T = 2.0",
    ]
    stress = stress[stress["setting"].isin(keep_settings)].copy()

    _plot_stress_overlay(
        stress, y="kappa_obs_mean",
        title=f"Stress z = {Z_STRESS}: κ over training — {MAIN_DATASET.upper()}\nThin: seeds; thick: mean",
        out_path=OUT_STRESS / "S1_kappa_over_training.png",
        logy=True, show_seed_lines=True, ylabel=YLABEL["kappa_obs_mean"]
    )
    _plot_stress_overlay(
        stress, y="attempts_p90",
        title=f"Stress z = {Z_STRESS}: sampling attempts (p90) over training — {MAIN_DATASET.upper()}\nThin: seeds; thick: mean",
        out_path=OUT_STRESS / "S2_attempts_p90_over_training.png",
        logy=True, show_seed_lines=True, ylabel=YLABEL["attempts_p90"]
    )
    _plot_stress_overlay(
        stress, y="train_seconds_per_batch",
        title=f"Stress z = {Z_STRESS}: Runtime per batch (s) over training over training — {MAIN_DATASET.upper()}\n{seed_note}; batch size = {BATCH_SIZE}",
        out_path=OUT_STRESS / "S3_runtime_seconds_per_batch.png",
        logy=False, show_seed_lines=True, ylabel=YLABEL["train_seconds_per_batch"]
    )
    _plot_stress_overlay(
        stress, y="Delta_sigmoid",
        title=f"Stress z = {Z_STRESS}: Δ over training — {MAIN_DATASET.upper()}\nThin: seeds; thick: mean",
        out_path=OUT_STRESS / "S4_delta_over_training.png",
        logy=False, show_seed_lines=True, ylabel=YLABEL["Delta_sigmoid"]
    )

    _final_epoch_summary_bar(
        stress,
        group_col="setting",
        metrics=["attempts_max", "train_seconds_per_batch"],
        title=f"Stress z = {Z_STRESS}: sampler tail and runtime — {MAIN_DATASET.upper()}",
        out_prefix=OUT_STRESS / "S5_final_epoch_bars",
        sort_by="attempts_max",
    )

    _final_epoch_summary_bar(
        stress,
        group_col="setting",
        metrics=["Delta_sigmoid", "LL"],
        title=f"Stress z = {Z_STRESS}: Δ and log-likelihood — {MAIN_DATASET.upper()}",
        out_prefix=OUT_STRESS / "S6_final_epoch_bars",
        sort_by="Delta_sigmoid",
    )

    _plot_scatter(
        stress, x="kappa_obs_mean", y="attempts_p90",
        title=f"Stress z = {Z_STRESS}: attempts (p90) vs κ — {MAIN_DATASET.upper()}",
        out_path=OUT_DIAG / "X1_attempts_p90_vs_kappa.png",
        logx=True, logy=True,
        xlabel="κ", ylabel=YLABEL["attempts_p90"]
    )
    _plot_scatter(
        stress, x="attempts_p90", y="train_seconds_per_batch",
        title=f"Stress z = {Z_STRESS}: runtime vs attempts (p90) — {MAIN_DATASET.upper()}",
        out_path=OUT_DIAG / "X2_runtime_vs_attempts_p90.png",
        logx=True, logy=False,
        xlabel=YLABEL["attempts_p90"], ylabel=YLABEL["train_seconds_per_batch"]
    )
    _plot_scatter(
        stress, x="attempts_p90", y="Delta_sigmoid",
        title=f"Stress z = {Z_STRESS}: Δ vs attempts (p90) — {MAIN_DATASET.upper()}",
        out_path=OUT_DIAG / "X3_delta_vs_attempts_p90.png",
        logx=True, logy=False,
        xlabel=YLABEL["attempts_p90"], ylabel=YLABEL["Delta_sigmoid"]
    )

# Appendix for sure: MNIST vs FashionMNIST (shared y-axis; vMF identity preserved)

def _make_setting(df_in):
    df2 = df_in.copy()
    df2["setting"] = df2.apply(_setting_label, axis=1)
    return df2

def plot_metric_mnist_vs_fmnist(df_vmf_all_in, z, metric, out_path, logy=True):
    if metric not in df_vmf_all_in.columns:
        return

    d = df_vmf_all_in[df_vmf_all_in["z_dim"] == z].copy()
    if len(d) == 0:
        return
    d = _make_setting(d)

    keep = {
        "clip = None, T = 1.0",
        "clip = None, T = 0.5",
        "clip = None, T = 2.0",
        "clip = 50, T = 1.0",
        "clip = 100, T = 1.0",
    }
    d = d[d["setting"].isin(keep)].copy()
    if len(d) == 0:
        return

    mean_df = d.groupby(["dataset", "setting", "epoch"], as_index=False)[metric].mean()
    max_ep = int(mean_df["epoch"].max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    marker = MODEL_STYLE["vmf"]["marker"]

    for ax, ds in zip(axes, ["mnist", "fashionmnist"]):
        sub = mean_df[mean_df["dataset"] == ds].copy()
        settings_sorted = sorted(sub["setting"].unique().tolist())
        pal = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"])
        cmap = {s: pal[i % len(pal)] for i, s in enumerate(settings_sorted)}

        for setting, g in sub.groupby("setting"):
            g = g.sort_values("epoch")
            ax.plot(
                g["epoch"], g[metric],
                linestyle=_stress_linestyle(setting),
                marker=marker,
                markersize=3.0,
                markevery=MARKEVERY_MEAN,
                linewidth=LW_MEAN,
                color=cmap.get(setting, None),
                label=setting,
            )
        prefix = "a) " if ds == "mnist" else "b) "
        ax.set_title(prefix + ds.upper())

        ax.set_xlabel("Epoch")
        _apply_style(ax, max_epoch=max_ep, logy=logy)

    axes[0].set_ylabel(YLABEL.get(metric, metric.replace("_", " ")))
    axes[1].legend(loc="best", fontsize=8, title="Setting")
    plt.suptitle(f"Appendix: {YLABEL.get(metric, metric.replace('_',' '))} — MNIST vs FashionMNIST (vMF, z = {z})\n{seed_note}")
    _savefig(out_path)

plot_metric_mnist_vs_fmnist(
    df_vmf_all, z=Z_STRESS, metric="kappa_obs_mean",
    out_path=OUT_APPX / "APPX_kappa_mean_mnist_vs_fmnist.png",
    logy=True
)

print("\nDone. Plots saved under:", OUT_DIR)
