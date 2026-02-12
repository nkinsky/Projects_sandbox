
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:47:07 2025
@author: Yoga

This script detects ripple events in hippocampal LFP data for three sessions
(saline_1, psilocybin, saline_2), computes ripple features (duration, peak power,
peak frequency, sharp-wave amplitude), compares them statistically, and exports
boxplots with p-values into a PDF file. The PDF is saved under:
    /Users/Yoga/Desktop/yeni_veri/<animal>/<animal>_ripple_feature_comparison_boxplots.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from statannotations.Annotator import Annotator
import getpass
from pathlib import Path

from neuropy.core import Epoch, ProcessData, Shank, Probe, ProbeGroup
from neuropy.utils.signal_process import filter_sig
from neuropy.analyses.oscillations import Ripple
from neuropy.analyses import oscillations
from Psilocybin.subjects import get_psi_dir

# Import method for getting data directories

if "kinsky" in getpass.getuser():
    get_dir = get_psi_dir
    def get_animal_dir(animal):
        return get_dir(animal, "saline1").parent
else:
    def get_dir(animal, recording):
        return os.path.join(BASE_DIR_ROOT, animal, recording)
    def get_animal_dir(animal):
        return os.path.join(BASE_DIR_ROOT, animal)

# -----------------------------
# Matplotlib config (Illustrator-friendly fonts)
# -----------------------------
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# -----------------------------
# User parameters
# -----------------------------
animal = "Finn2"
ripple_thresh = (4, None)  # lower thresh, upper_thresh
animal = animal.lower()
recordings = ["saline_1", "psilocybin", "saline_2"]

# Root folder that contains per-animal/session data
BASE_DIR_ROOT = "/data3/Psilocybin/Recording_Rats"

# Set how you want to deal with different lengths between saline and psilocybin sessions
# (Finn, Rey, Rose all had ~1hr saline and ~4hr Psilocybin, shorter for Rose though due to disconnects)

# Cut down Finn2 saline to 1hr?
chop_finn2_saline = False  # True = only use 1st hour of Finn2 saline, False = use all
finn2_append = "_1hrsalineonly" if chop_finn2_saline else ""

# ... OR only use 1hr Psilocybin for all
limit_to_1st_hr = True
chop_all_append = "_allsessions1hr" if limit_to_1st_hr else ""
finn2_append = "" if chop_all_append else finn2_append

# -----------------------------
# Channel map per animal and session
# -----------------------------
chan_dict = {
    "rey":   {"saline_1": 21, "psilocybin": 21, "saline_2": 22},
    "finn":  {"saline_1": 27, "psilocybin": 27, "saline_2": 27},
    "rose":  {"saline_1": 27, "psilocybin": 26, "saline_2": 25},
    "finn2": {"saline_1": 4,  "psilocybin": 4,  "saline_2": 4}
}

# Labels for plotting
recording_labels = {
    "saline_1": "Saline 1",
    "psilocybin": "Psilocybin",
    "saline_2": "Saline 2"
}

# -----------------------------
# Container for ripple data across sessions
# -----------------------------
all_ripple_data = []

# Ensure animal directory exists for outputs
animal_dir = get_animal_dir(animal)
os.makedirs(animal_dir, exist_ok=True)

# -----------------------------
# Process each session
# -----------------------------
for recording in recordings:
    print(f"Processing: {animal} - {recording}")
    ripple_kanali = chan_dict[animal][recording]

    # Session directory
    # diruse = os.path.join(BASE_DIR_ROOT, animal, recording)
    diruse = get_dir(animal, recording)
    print(diruse)
    if not os.path.isdir(diruse):
        raise FileNotFoundError(f"Session directory not found: {diruse}")

    # Load session with neuropy
    sess = ProcessData(diruse)
    signal = sess.eegfile.get_signal()

    # Only include first hour for Finn2
    if limit_to_1st_hr | (chop_finn2_saline & (animal == "finn2") & ("saline" in recording)):
        signal = signal.time_slice(t_start=0, t_stop=np.min((3600, signal.t_stop)))

    # Load artifact file (contains artifact epochs)
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

    # Define probe and shank structure
    # shank = Shank().auto_generate(columns=1, contacts_per_column=32,
    #                               xpitch=0, ypitch=50,
    #                               channel_id=np.arange(32, 0, -1))

    # NRK bugfix start
    shank = Shank().auto_generate(columns=1, contacts_per_column=32,
                                  xpitch=0, ypitch=-50,
                                  channel_id=np.arange( 0, 32, 1))
    # NRK bugfix end

    shank.set_disconnected_channels(sess.recinfo.skipped_channels)
    probe = Probe(shank)
    prbgrp = ProbeGroup()
    prbgrp.add_probe(probe)
    sess.prbgrp = prbgrp

    # Set detection threshold (higher for psilocybin)
    # ripple_thresh = (4, None) if recording == "psilocybin" else (2.5, None)

    # Ripple detection, excluding artifact epochs
    ripple_epochs, ripple_power = oscillations.detect_ripple_epochs(
        signal, sess.prbgrp,
        thresh=ripple_thresh,
        ripple_channel=ripple_kanali,
        ignore_epochs=art_epochs,
        mindur=0.05,
        return_power=True
    )

    # Compute peak ripple frequency
    # NRK bugfix - this should not be necessary, should just be able to use ripple_kanali...
    # Map the absolute channel id to the index within prbgrp
    try:
        # NRK comment: this is not necessarily a bug with the fixed ProbeGroup code above,
        # but it was the source of the Rose issues. As originally written, the probegroup channels
        # were ordered [31, 30, 29 ..., 2, 1]. For the psilocybin session, the ripple_kanali of 26
        # would then retrun rpl_indx = 6, and channel 6 is a bad channel. That's why all the ripple
        # power and frequency calculations were off.  You should be able to run this all using just
        # the code under the Exception block below, using just the ripple_kanali directly in
        # Ripple_get_peak_ripple_freq() below.
        rpl_idx = np.where(ripple_kanali == sess.prbgrp.channel_id)[0][0]
    except Exception:
        # Fallback to same id if channel_id is already an index
        rpl_idx = ripple_kanali
    print([f"rpl_idx={rpl_idx}"])

    ripple_epochs2 = Ripple.get_peak_ripple_freq(
        sess.eegfile,
        rpl_channel=rpl_idx,
        rpl_epochs=ripple_epochs
    )
    rpl_df = ripple_epochs2.to_dataframe()  # columns include: duration, peak_power, peak_frequency_bp, etc.

    # -----------------------------
    # Compute sharp-wave amplitude
    # -----------------------------
    # Helper to extract the extreme value (by absolute magnitude)
    get_extrema = lambda arr: arr[np.argmax(np.abs(arr))]

    def get_max_val(x, t, bins):
        # Returns binned extreme values; then every other bin (start/stop binning)
        return stats.binned_statistic(t, x, bins=bins, statistic=get_extrema)[0][::2]

    srate = sess.eegfile.sampling_rate
    rpl_epochs_flat = ripple_epochs2.flatten()
    rpl_traces, t = sess.eegfile.get_frames_within_epochs(
        ripple_epochs2,
        np.arange(sess.eegfile.n_channels),
        ret_time=True
    )
    rpl_traces = filter_sig.bandpass(rpl_traces, lf=2, hf=30, fs=srate)

    # Parallel processing across channels for efficiency
    max_val = Parallel(n_jobs=8)(
        delayed(get_max_val)(arr, t, rpl_epochs_flat) for arr in rpl_traces
    )
    max_val = np.asarray(max_val)

    # Peak-to-peak amplitude (mV scale factor)
    sharp_wave_ampls = np.ptp(max_val, axis=0) * 0.95 * 1e-3

    # Append to main DataFrame
    rpl_df["sharp_wave_amplitude"] = sharp_wave_ampls
    rpl_df["session"] = recording_labels[recording]
    all_ripple_data.append(rpl_df)

# -----------------------------
# Combine all sessions
# -----------------------------
full_df = pd.concat(all_ripple_data, ignore_index=True)
full_df.to_csv(animal_dir / "aggdata"/ f"{animal}_rpl_features_thresh{'_'.join(str(ripple_thresh[0]).split('.'))}{chop_all_append}.csv")

# -----------------------------
# Features and titles for plotting
# -----------------------------
features = ["duration", "peak_power", "sharp_wave_amplitude", "peak_frequency_bp"]
feature_titles = {
    "duration": "Ripple Duration (s)",
    "peak_power": "Ripple Peak Power",
    "sharp_wave_amplitude": "Sharp Wave Amplitude (mV)",
    "peak_frequency_bp": "Ripple Peak Frequency (Hz)"
}

# -----------------------------
# Generate PDF with boxplots and statistical tests
# -----------------------------
pdf_path = os.path.join(animal_dir, f"{animal}_ripple_feature_comparison_boxplots_ripplethresh{'_'.join(str(ripple_thresh[0]).split('.'))}{finn2_append}{chop_all_append}.pdf")

with PdfPages(pdf_path) as pdf:
    for feature in features:
        plot_df = full_df[["session", feature]].dropna().rename(columns={feature: "value"})

        # Boxplot + Stripplot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=plot_df, x="session", y="value",
            showfliers=False, boxprops=dict(alpha=0.6), ax=ax
        )
        sns.stripplot(
            data=plot_df, x="session", y="value",
            color='black', size=4, alpha=0.6, jitter=0.2, ax=ax
        )

        # Define pairs for pairwise t-tests
        order = ["Saline 1", "Psilocybin", "Saline 2"]
        pairs = [("Saline 1", "Psilocybin"),
                 ("Psilocybin", "Saline 2"),
                 ("Saline 1", "Saline 2")]

        annotator = Annotator(ax, pairs, data=plot_df, x="session", y="value", order=order)
        annotator.configure(test='t-test_ind', text_format='star', loc='outside')
        annotator.apply_and_annotate()

        # Display numeric p-values as text
        p_text_lines = []
        for pair in pairs:
            g1 = plot_df[plot_df["session"] == pair[0]]["value"]
            g2 = plot_df[plot_df["session"] == pair[1]]["value"]
            _, pval = stats.ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
            p_text_lines.append(f"{pair[0]} vs {pair[1]}: p = {pval:.4f}")
        p_text = "\n".join(p_text_lines)

        ax.set_title(f"{animal.capitalize()} – {feature_titles[feature]}", fontsize=13)
        ax.set_xlabel("Session")
        ax.set_ylabel(feature_titles[feature])
        ax.text(0.5, -0.3, p_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9)

        if feature in ["peak_power", "sharp_wave_amplitude"]:  # zoom in to sw_amplitude
            for zoom_in in [False, True]:
                if zoom_in:
                    quartiles = np.quantile(plot_df["value"].values, (0.25, 0.75))
                    IQR = np.diff(quartiles)[0]
                    ylims = (quartiles[0] - 2 * IQR, quartiles[1] + 2 * IQR)
                    ax.set_ylim(ylims)
                    ax.set_title(f"{animal.capitalize()} – {feature_titles[feature]}: zoomed", fontsize=13)

                sns.despine(ax=ax)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        else:
            sns.despine(ax=ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

print(f"\nPDF successfully created: {pdf_path}")

if __name__ == "__main__":
    pass
