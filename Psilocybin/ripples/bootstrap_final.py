
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 12:09:02 2025

@author: Yoga

Group-level ripple feature analysis with cluster-level (animal) bootstrap
+ Ca2+-ripple coupling (peri-ripple Ca response)
+ session-level ripple rates, frequencies, Ca2+ correlations.

Outputs:

1) Event-level features (duration, power, SW amplitude, peak frequency, peri-ripple Ca):
    

2) Session-level metrics (ripple rate, mean frequency, Ca correlation):
   
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages

from neuropy.core import Epoch, ProcessData, Shank, Probe, ProbeGroup
from neuropy.utils.signal_process import filter_sig
from neuropy.analyses.oscillations import Ripple
from neuropy.analyses import oscillations


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


animals = ["finn", "rey", "rose", "finn2"]
recordings = ["saline_1", "psilocybin", "saline_2"]


BASE_DIR_ROOT = "/Users/Yoga/Desktop/data_psilocybin"

# Ca2+ data root and options 
# USE_CA_COUPLING: whether to compute Ca2+-ripple coupling
# CA_WINDOW: time window (s) around ripple peak for peri-ripple Ca calculation
# CA_FPS: Ca imaging frame rate (Hz) <-- update to the true FPS
USE_CA_COUPLING = True
CA_BASE_DIR_ROOT = "/Users/Yoga/Desktop/ca_image"
CA_WINDOW = (-0.5, 0.5)
CA_FPS = 20.0


CA_DIR_MAP = {
    "rey": {
        "saline_1": "2022_06_01_saline1",
        "psilocybin": "2022_06_02_psilocybin",
        "saline_2": "2022_06_03_saline2",
    },
    "finn": {
        "saline_1": "2022_02_15_saline1",
        "psilocybin": "2022_02_17_psilocybin",
        "saline_2": "2022_02_18_saline2",
    },
    "finn2": {
        "saline_1": "2023_05_24_saline1",
        "psilocybin": "2023_05_25_psilocybin",
        "saline_2": "2023_05_26_saline2",
    },
}

SESSION_DURATION_MAP = {
    }
# Bootstrap settings (cluster-level, animals)
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

# Set to False if you want to disable sharp-wave amplitude computation
USE_SHARP_WAVE = True

OVERLAY_ANIMAL_MEANS_ON_EVENT_PLOTS = True
CONNECT_ANIMAL_MEANS_ACROSS_SESSIONS = True
ANIMAL_MEAN_MARKER_SIZE = 80
ANIMAL_MEAN_LINE_ALPHA = 0.6
ANIMAL_MEAN_JITTER_WIDTH = 0.25

chan_dict = {
    "rey":   {"saline_1": 21, "psilocybin": 21, "saline_2": 22},
    "finn":  {"saline_1": 27, "psilocybin": 27, "saline_2": 27},
    "rose":  {"saline_1": 27, "psilocybin": 26, "saline_2": 25},
    "finn2": {"saline_1": 4,  "psilocybin": 4,  "saline_2": 4}
}


recording_labels = {
    "saline_1": "Saline 1",
    "psilocybin": "Psilocybin",
    "saline_2": "Saline 2"
}


# Find Ca session folder and load C.npy / C_<session>.npy


def find_ca_session_dir(animal, recording):
    """
    Find the Ca imaging session folder using CA_DIR_MAP.
    Example: Rey + saline_1 -> ca_image/Rey/2022_06_01_saline1
    """
    animal_lower = animal.lower()
    animal_cap = animal.capitalize()

    if animal_lower not in CA_DIR_MAP or recording not in CA_DIR_MAP[animal_lower]:
        raise FileNotFoundError(
            f"No Ca folder mapping defined for {animal} - {recording} in CA_DIR_MAP."
        )

    folder_name = CA_DIR_MAP[animal_lower][recording]
    sess_dir = os.path.join(CA_BASE_DIR_ROOT, animal_cap, folder_name)

    if not os.path.isdir(sess_dir):
        raise FileNotFoundError(f"Ca session directory does not exist: {sess_dir}")

    return sess_dir


def load_ca_data(animal, recording):
    
    sess_dir = find_ca_session_dir(animal, recording)

    candidates = [
        f for f in os.listdir(sess_dir)
        if f.lower().endswith(".npy") and (f.lower().startswith("c_") or f.lower() == "c.npy")
    ]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No C_*.npy or C.npy file found in {sess_dir} for {animal} - {recording}"
        )
    if len(candidates) > 1:
        print(f"[WARN] Multiple C files found in {sess_dir}: {candidates}. Using first.")

    c_path = os.path.join(sess_dir, candidates[0])

    C_raw = np.load(c_path, allow_pickle=True)

    if C_raw.dtype == object:
        try:
            C_float = np.array(C_raw, dtype=float)
        except Exception:
            C_float = np.vstack([np.asarray(row, dtype=float) for row in C_raw])
    else:
        C_float = np.asarray(C_raw, dtype=float)


    n0, n1 = C_float.shape
    ca_traces = C_float if n0 <= n1 else C_float.T

    n_cells, n_frames = ca_traces.shape

  
    ca_time = np.arange(n_frames, dtype=float) / CA_FPS

    return ca_traces, ca_time


def compute_peri_ripple_ca_mean(ca_traces, ca_time, rpl_df, window=CA_WINDOW):
    """
    For each ripple, compute the mean Ca activity (ΔF/F) within a specified time window
    around the ripple peak.

    """
    t_start_rel, t_stop_rel = window
    ca_ripple_mean = []

    if "peak_time" in rpl_df.columns:
        ripple_times = rpl_df["peak_time"].values
    else:
        ripple_times = (rpl_df["start"].values + rpl_df["stop"].values) / 2.0

    for t_peak in ripple_times:
        t_start = t_peak + t_start_rel
        t_stop = t_peak + t_stop_rel
        mask = (ca_time >= t_start) & (ca_time <= t_stop)

        if not np.any(mask):
            ca_ripple_mean.append(np.nan)
        else:
            ca_ripple_mean.append(np.nanmean(ca_traces[:, mask]))

    return np.array(ca_ripple_mean)



# cluster-level bootstrap (clusters = animals)

def cluster_bootstrap_diff(
    summary_df,
    group_col,
    value_col,
    group1,
    group2,
    cluster_col="animal",
    n_boot=2000,
    random_state=None
):
    """
    Cluster-level bootstrap (cluster = animal).

    This compares two sessions by:
    - computing the mean per animal within each session,
    - bootstrapping animals (with replacement),
    - estimating the distribution of mean differences (group2 - group1),
    - returning observed difference, two-sided bootstrap p-value, and 95% CI.
    """
    rng = np.random.default_rng(random_state)

    sub = summary_df[summary_df[group_col].isin([group1, group2])].copy()
    sub = sub[[group_col, value_col, cluster_col]].dropna()

    obs_g1 = sub[sub[group_col] == group1].groupby(cluster_col)[value_col].mean()
    obs_g2 = sub[sub[group_col] == group2].groupby(cluster_col)[value_col].mean()
    obs_diff = obs_g2.mean() - obs_g1.mean()

    clusters = sub[cluster_col].unique()
    n_clusters = len(clusters)
    boot_diffs = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)

        boot_g1 = []
        boot_g2 = []

        for cl in sampled_clusters:
            cl_vals_g1 = sub[(sub[group_col] == group1) & (sub[cluster_col] == cl)][value_col].values
            cl_vals_g2 = sub[(sub[group_col] == group2) & (sub[cluster_col] == cl)][value_col].values

            if len(cl_vals_g1) == 0 or len(cl_vals_g2) == 0:
                continue

            boot_g1.append(cl_vals_g1.mean())
            boot_g2.append(cl_vals_g2.mean())

        if len(boot_g1) == 0 or len(boot_g2) == 0:
            boot_diffs[i] = 0.0
        else:
            boot_diffs[i] = np.mean(boot_g2) - np.mean(boot_g1)

    p_greater = np.mean(boot_diffs >= 0)
    p_less = np.mean(boot_diffs <= 0)
    p_two_sided = 2 * min(p_greater, p_less)

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    return obs_diff, p_two_sided, ci_low, ci_high, boot_diffs


def p_to_stars(p):
   
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

all_ripple_data = []      # event-level
session_metrics_rows = [] # session-level (rate, mean freq, Ca corr)


for animal in animals:
    print(f"\n=== Processing animal: {animal} ===")

    animal_dir = os.path.join(BASE_DIR_ROOT, animal)
    os.makedirs(animal_dir, exist_ok=True)

    for recording in recordings:
        print(f"  Session: {recording}")
        ripple_kanali = chan_dict[animal][recording]

        # LFP session directory
        diruse = os.path.join(BASE_DIR_ROOT, animal, recording)
        if not os.path.isdir(diruse):
            raise FileNotFoundError(f"Session directory not found: {diruse}")

    
        sess = ProcessData(diruse)

        
        signal = sess.eegfile.get_signal()
        srate = sess.eegfile.sampling_rate

        animal_cap = animal.capitalize()
        recording_clean = recording.replace("_", "")
        artifact_file = os.path.join(diruse, f"{animal_cap}_{recording_clean}_denoised.artifact.npy")

        if not os.path.exists(artifact_file):
            raise FileNotFoundError(f"Artifact file not found: {artifact_file}")

        artifact_obj = np.load(artifact_file, allow_pickle=True).item()
        epochs_dict = artifact_obj.get("epochs", {})
        start_times = [epochs_dict["start"][k] for k in sorted(epochs_dict["start"].keys())]
        stop_times = [epochs_dict["stop"][k] for k in sorted(epochs_dict["stop"].keys())]
        art_epochs = Epoch(pd.DataFrame({
            "start": start_times,
            "stop": stop_times,
            "label": ["artifact"] * len(start_times)
        }))

        
        shank = Shank().auto_generate(columns=1, contacts_per_column=32,
                                      xpitch=0, ypitch=50,
                                      channel_id=np.arange(32, 0, -1))
        shank.set_disconnected_channels(sess.recinfo.skipped_channels)
        probe = Probe(shank)
        prbgrp = ProbeGroup()
        prbgrp.add_probe(probe)
        sess.prbgrp = prbgrp

        
        ripple_thresh = (4, None) if recording == "psilocybin" else (2.5, None)

       
        ripple_epochs, ripple_power = oscillations.detect_ripple_epochs(
            signal, sess.prbgrp,
            thresh=ripple_thresh,
            ripple_channel=ripple_kanali,
            ignore_epochs=art_epochs,
            mindur=0.05,
            return_power=True
        )

        
        try:
            rpl_idx = np.where(ripple_kanali == sess.prbgrp.channel_id)[0][0]
        except Exception:
            rpl_idx = ripple_kanali

        ripple_epochs2 = Ripple.get_peak_ripple_freq(
            sess.eegfile,
            rpl_channel=rpl_idx,
            rpl_epochs=ripple_epochs
        )
        rpl_df = ripple_epochs2.to_dataframe()  # duration, peak_power, peak_frequency_bp, peak_time, ...

        
        if USE_SHARP_WAVE and len(ripple_epochs) > 0:
            get_extrema = lambda arr: arr[np.argmax(np.abs(arr))]

            def get_max_val(x, t, bins):
                return stats.binned_statistic(t, x, bins=bins, statistic=get_extrema)[0][::2]

            rpl_epochs_flat = ripple_epochs.flatten()
            rpl_traces, t = sess.eegfile.get_frames_within_epochs(
                ripple_epochs,
                np.arange(sess.eegfile.n_channels),
                ret_time=True
            )
            rpl_traces = filter_sig.bandpass(rpl_traces, lf=2, hf=30, fs=srate)

            max_val = Parallel(n_jobs=8)(
                delayed(get_max_val)(arr, t, rpl_epochs_flat) for arr in rpl_traces
            )
            max_val = np.asarray(max_val)

            sharp_wave_ampls = np.ptp(max_val, axis=0) * 0.95 * 1e-3
            rpl_df["sharp_wave_amplitude"] = sharp_wave_ampls
        else:
            rpl_df["sharp_wave_amplitude"] = np.nan

       
        if USE_CA_COUPLING and len(rpl_df) > 0:
            try:
                ca_traces, ca_time = load_ca_data(animal, recording)
                ca_ripple_mean = compute_peri_ripple_ca_mean(
                    ca_traces, ca_time, rpl_df, window=CA_WINDOW
                )
                rpl_df["ca_ripple_mean"] = ca_ripple_mean
            except FileNotFoundError as e:
                print(f"[WARN] Ca2+ data missing for {animal} - {recording}: {e}")
                rpl_df["ca_ripple_mean"] = np.nan
        else:
            rpl_df["ca_ripple_mean"] = np.nan

        # Metadata columns
        rpl_df["session"] = recording_labels[recording]
        rpl_df["animal"] = animal

        all_ripple_data.append(rpl_df)

      
        if animal in SESSION_DURATION_MAP and recording in SESSION_DURATION_MAP[animal]:
            session_duration_s = SESSION_DURATION_MAP[animal][recording]
        else:
            session_duration_s = None

            if hasattr(sess.eegfile, "duration"):
                try:
                    session_duration_s = float(sess.eegfile.duration)
                except Exception:
                    session_duration_s = None

            if session_duration_s is None and hasattr(signal, "shape") and signal.shape is not None:
                shp = signal.shape
                if len(shp) == 2:
                    n_samples = shp[1]
                    session_duration_s = n_samples / srate

            if session_duration_s is None:
                if len(ripple_epochs) > 0:
                    try:
                        approx_dur = float(ripple_epochs.flatten().max())
                        session_duration_s = approx_dur
                        print(f"[WARN] Approximate session duration used for {animal} - {recording}: "
                              f"{approx_dur:.1f} s (from ripple epochs)")
                    except Exception:
                        session_duration_s = np.nan
                        print(f"[WARN] Could not determine session duration for {animal} - {recording}. "
                              f"Ripple rate will be NaN.")
                else:
                    session_duration_s = np.nan
                    print(f"[WARN] No ripples and no duration info for {animal} - {recording}. "
                          f"Ripple rate will be NaN.")

        # ---- Session-level metrics ----
        # ripple_rate_per_min: number of ripples per minute
        # mean_peak_frequency: mean of peak ripple frequencies (per session)
        # ca_ripple_corr: correlation between ripple duration and peri-ripple Ca mean (requires enough valid points)
        n_ripples = len(rpl_df)
        if session_duration_s is not None and session_duration_s > 0 and not np.isnan(session_duration_s):
            ripple_rate_per_min = (n_ripples / session_duration_s) * 60.0
        else:
            ripple_rate_per_min = np.nan

        mean_peak_frequency = rpl_df["peak_frequency_bp"].mean() if "peak_frequency_bp" in rpl_df.columns else np.nan

        if USE_CA_COUPLING and "ca_ripple_mean" in rpl_df.columns:
            valid = rpl_df[["duration", "ca_ripple_mean"]].dropna()
            if len(valid) >= 5:
                ca_ripple_corr = np.corrcoef(valid["duration"], valid["ca_ripple_mean"])[0, 1]
            else:
                ca_ripple_corr = np.nan
        else:
            ca_ripple_corr = np.nan

        session_metrics_rows.append({
            "animal": animal,
            "session": recording_labels[recording],
            "ripple_rate_per_min": ripple_rate_per_min,
            "mean_peak_frequency": mean_peak_frequency,
            "ca_ripple_corr": ca_ripple_corr
        })


full_df = pd.concat(all_ripple_data, ignore_index=True)
session_metrics_df = pd.DataFrame(session_metrics_rows)

# -----------------------------
# 1) Event-level features (PDF 1)
# -----------------------------
# For each feature:
# Boxplot summarizes the distribution across all ripple events per session.
# Stripplot shows all ripple events, colored by animal.
# Optional overlay: animal-by-session mean markers (diamonds) and connecting lines across sessions.
# Cluster-level bootstrap statistics are computed using animal means (one mean per animal per session).
features = [
    "duration",
    "peak_power",
    "sharp_wave_amplitude",
    "peak_frequency_bp",
    "ca_ripple_mean"
]
feature_titles = {
    "duration": "Ripple Duration (s)",
    "peak_power": "Ripple Peak Power",
    "sharp_wave_amplitude": "Sharp Wave Amplitude (mV)",
    "peak_frequency_bp": "Ripple Peak Frequency (Hz)",
    "ca_ripple_mean": "Mean Peri-ripple Ca²⁺ Response (ΔF/F)"
}

order = ["Saline 1", "Psilocybin", "Saline 2"]
pairs = [("Saline 1", "Psilocybin"),
         ("Psilocybin", "Saline 2"),
         ("Saline 1", "Saline 2")]

# Build animal × session summary table (used for cluster-level bootstrap)
summary_rows = []
for feat in features:
    for (animal, session), grp in full_df.groupby(["animal", "session"]):
        mean_val = grp[feat].mean()
        summary_rows.append({
            "animal": animal,
            "session": session,
            "feature": feat,
            "mean_val": mean_val
        })

summary_df = pd.DataFrame(summary_rows)

group_pdf_path = os.path.join(BASE_DIR_ROOT, "group_level_ripple_feature_bootstrap_boxplots_fast.pdf")

# Helper mapping for manual x-offsets per animal (for mean overlay)
animal_order = animals[:]  # preserve user-defined order
offsets = np.linspace(-ANIMAL_MEAN_JITTER_WIDTH, ANIMAL_MEAN_JITTER_WIDTH, len(animal_order))
animal_to_offset = {a: off for a, off in zip(animal_order, offsets)}

with PdfPages(group_pdf_path) as pdf:
    for feature in features:
        plot_df = full_df[["session", "animal", feature]].dropna().rename(columns={feature: "value"})

        if plot_df.empty:
            print(f"Warning: feature {feature} has no valid data, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        # Distribution summary (boxplot)
        sns.boxplot(
            data=plot_df, x="session", y="value",
            order=order,
            showfliers=False, boxprops=dict(alpha=0.6), ax=ax
        )

        # All event-level data points (stripplot)
        sns.stripplot(
            data=plot_df, x="session", y="value",
            order=order,
            hue="animal",
            dodge=True,
            size=3, alpha=0.6, ax=ax
        )

        # Legend for animals (from stripplot)
        ax.legend(title="Animal", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

        
        if OVERLAY_ANIMAL_MEANS_ON_EVENT_PLOTS:
            mean_df = (plot_df
                       .groupby(["animal", "session"], as_index=False)["value"]
                       .mean()
                       .rename(columns={"value": "animal_mean"}))

            for animal in animal_order:
                adf = mean_df[mean_df["animal"] == animal].copy()
                if adf.empty:
                    continue

                xs, ys = [], []
                for sess_name in order:
                    row = adf[adf["session"] == sess_name]
                    if row.empty:
                        continue

                    x = order.index(sess_name) + animal_to_offset.get(animal, 0.0)
                    y = float(row["animal_mean"].values[0])
                    xs.append(x)
                    ys.append(y)

                    ax.scatter(
                        x, y,
                        s=ANIMAL_MEAN_MARKER_SIZE,
                        marker="D",  # diamond marker for animal means
                        edgecolor="k",
                        linewidth=0.8,
                        zorder=5
                    )

                if CONNECT_ANIMAL_MEANS_ACROSS_SESSIONS and len(xs) >= 2:
                    ax.plot(xs, ys, linewidth=1.2, alpha=ANIMAL_MEAN_LINE_ALPHA, zorder=4)

            ax.text(0.99, 0.02, "Diamond = animal mean (per session)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

        feat_summary = summary_df[summary_df["feature"] == feature].copy()

        # Bootstrap p-text + significance stars
        p_text_lines = []
        y_max = plot_df["value"].max()
        y_min = plot_df["value"].min()
        y_range = y_max - y_min if y_max > y_min else 1.0
        base_y = y_max + 0.10 * y_range
        h = 0.06 * y_range

        for i, (g1, g2) in enumerate(pairs):
            obs_diff, p_boot, ci_low, ci_high, _boot_diffs = cluster_bootstrap_diff(
                feat_summary,
                group_col="session",
                value_col="mean_val",
                group1=g1,
                group2=g2,
                cluster_col="animal",
                n_boot=N_BOOTSTRAP,
                random_state=RANDOM_SEED + i
            )

            stars = p_to_stars(p_boot)
            p_text_lines.append(
                f"{g1} vs {g2}: Δmean = {obs_diff:.4f}, "
                f"p_boot = {p_boot:.4f} ({stars}), "
                f"95% CI [{ci_low:.4f}, {ci_high:.4f}]"
            )

            x1 = order.index(g1)
            x2 = order.index(g2)
            x_center = (x1 + x2) / 2
            y = base_y + i * h

            ax.plot([x1, x1, x2, x2],
                    [y, y + h * 0.2, y + h * 0.2, y],
                    color='k', linewidth=1)
            ax.text(x_center, y + h * 0.25, stars, ha='center', va='bottom', fontsize=11)

        p_text = "\n".join(p_text_lines)

        ax.set_title(f"Group-level (cluster=animal) – {feature_titles[feature]}", fontsize=13)
        ax.set_xlabel("Session")
        ax.set_ylabel(feature_titles[feature])
        ax.text(0.5, -0.35, p_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9)

        sns.despine(ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"\n[1] Group-level feature PDF created:\n{group_pdf_path}")

# -----------------------------
# 2) Session-level metrics (PDF 2)
# -----------------------------
# These plots use one value per animal per session:
# - ripple rate
# - mean peak frequency
# - Ca coupling correlation
#
# Boxplot summarizes across animals; stripplot shows each animal’s single session value.
# Bootstrap statistics again use animal as the cluster.
session_features = {
    "ripple_rate_per_min": "Ripple Rate (events/min)",
    "mean_peak_frequency": "Mean Ripple Peak Frequency (Hz)",
    "ca_ripple_corr": "Corr(duration, peri-ripple Ca²⁺)"
}

ratecorr_pdf_path = os.path.join(BASE_DIR_ROOT, "group_level_ripple_rate_freq_caCorr_bootstrap.pdf")

ratecorr_rows = []
for _, row in session_metrics_df.iterrows():
    for feat_name in session_features.keys():
        ratecorr_rows.append({
            "animal": row["animal"],
            "session": row["session"],
            "feature": feat_name,
            "mean_val": row[feat_name]
        })

summary_ratecorr = pd.DataFrame(ratecorr_rows)

with PdfPages(ratecorr_pdf_path) as pdf:
    for feat_name, feat_title in session_features.items():
        feat_df = summary_ratecorr[summary_ratecorr["feature"] == feat_name].dropna()
        if feat_df.empty:
            print(f"Warning: session-level feature {feat_name} has no valid data, skipping.")
            continue

        plot_df = feat_df[["session", "animal", "mean_val"]].copy()
        plot_df = plot_df.rename(columns={"mean_val": "value"})

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            data=plot_df, x="session", y="value",
            order=order,
            showfliers=False, boxprops=dict(alpha=0.6), ax=ax
        )
        sns.stripplot(
            data=plot_df, x="session", y="value",
            order=order,
            hue="animal",
            dodge=True,
            size=6, alpha=0.8, ax=ax
        )
        ax.legend(title="Animal", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

        p_text_lines = []
        y_max = plot_df["value"].max()
        y_min = plot_df["value"].min()
        y_range = y_max - y_min if y_max > y_min else 1.0
        base_y = y_max + 0.10 * y_range
        h = 0.06 * y_range

        for i, (g1, g2) in enumerate(pairs):
            obs_diff, p_boot, ci_low, ci_high, _boot_diffs = cluster_bootstrap_diff(
                feat_df,
                group_col="session",
                value_col="mean_val",
                group1=g1,
                group2=g2,
                cluster_col="animal",
                n_boot=N_BOOTSTRAP,
                random_state=RANDOM_SEED + 100 + i
            )

            stars = p_to_stars(p_boot)
            p_text_lines.append(
                f"{g1} vs {g2}: Δmean = {obs_diff:.4f}, "
                f"p_boot = {p_boot:.4f} ({stars}), "
                f"95% CI [{ci_low:.4f}, {ci_high:.4f}]"
            )

            x1 = order.index(g1)
            x2 = order.index(g2)
            x_center = (x1 + x2) / 2
            y = base_y + i * h

            ax.plot([x1, x1, x2, x2],
                    [y, y + h * 0.2, y + h * 0.2, y],
                    color='k', linewidth=1)
            ax.text(x_center, y + h * 0.25, stars, ha='center', va='bottom', fontsize=11)

        p_text = "\n".join(p_text_lines)

        ax.set_title(f"Group-level (cluster=animal) – {feat_title}", fontsize=13)
        ax.set_xlabel("Session")
        ax.set_ylabel(feat_title)
        ax.text(0.5, -0.35, p_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9)

        sns.despine(ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"\n[2] Session-level rate/freq/CaCorr PDF created:\n{ratecorr_pdf_path}")
