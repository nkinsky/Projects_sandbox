
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 2026
@author: Yoga

Unified SINGLE-PANEL per feature (one plot per ripple metric)
X: Animal 1..4, Hue: Session (S, P, S)
Hollow boxplot + stripplot + MWU stars (per animal within-session comparisons)
Saved into one multi-page PDF.
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
import getpass # NRK compatibility addition

from neuropy.core import Epoch, ProcessData, Shank, Probe, ProbeGroup
from neuropy.utils.signal_process import filter_sig
from neuropy.analyses.oscillations import Ripple
from neuropy.analyses import oscillations
from Psilocybin.subjects import get_psi_dir  # NRK compatiblity addition

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set_style("white", {'axes.edgecolor': 'black', 'axes.linewidth': 0.6})
sns.set_context("paper", font_scale=1.0)


animals = ["rey", "finn", "rose", "finn2"]
recordings = ["saline_1", "psilocybin", "saline_2"]
FIXED_THRESHOLD = 4.0
BASE_DIR_ROOT = "/Users/Yoga/Desktop/data_psilocybin"

# Import method for getting data directories - NRK compatibility
if "kinsky" in getpass.getuser():
    get_dir = get_psi_dir
    def get_animal_dir(animal):
        return get_dir(animal, "saline1").parent
else:
    def get_dir(animal, recording):
        return os.path.join(BASE_DIR_ROOT, animal, recording)
    def get_animal_dir(animal):
        return os.path.join(BASE_DIR_ROOT, animal)

chan_dict = {
    "rey":   {"saline_1": 21, "psilocybin": 21, "saline_2": 21},
    "finn":  {"saline_1": 27, "psilocybin": 27, "saline_2": 27},
    "rose":  {"saline_1": 26, "psilocybin": 26, "saline_2": 26},
    "finn2": {"saline_1": 4,  "psilocybin": 4,  "saline_2": 4}
}

recording_labels = {"saline_1": "Saline 1", "psilocybin": "Psilocybin", "saline_2": "Saline 2"}


order_sessions = ["Saline 1", "Psilocybin", "Saline 2"]
x_tick_labels_animals = ["Animal 1", "Animal 2", "Animal 3", "Animal 4"]


palette = {"Saline 1": "#1f77b4", "Psilocybin": "#ff7f0e", "Saline 2": "#2ca02c"}

group_ripple_data = []

def load_artifact_epochs(diruse, animal_cap, recording_clean):
    artifact_csv = os.path.join(diruse, f"{animal_cap}_{recording_clean}_denoised.artifact.csv")
    if os.path.exists(artifact_csv):
        try:
            tmp = pd.read_csv(artifact_csv)
            start_col = next((c for c in tmp.columns if 'start' in c.lower()), None)
            stop_col  = next((c for c in tmp.columns if ('stop' in c.lower()) or ('end' in c.lower())), None)
            if start_col and stop_col:
                art_df = pd.DataFrame({
                    "start": pd.to_numeric(tmp[start_col], errors="coerce"),
                    "stop":  pd.to_numeric(tmp[stop_col],  errors="coerce"),
                    "label": "artifact"
                }).dropna()
                if not art_df.empty:
                    return Epoch(art_df)
        except Exception:  # NRK comment: In general you do NOT want to use any exception here because you can end
            # up catching bugs that you might not have anticipated. You want to use a specific type of error here,
            # e.g. 'FileNotFoundError' if you are worried about missing a file, or 'TypeError' if you are worried that
            # some files are not being imported to the correct data type
            pass
    return None


# Data Processing

for animal in animals:
    animal_cap = animal.capitalize()
    for recording in recordings:
        print(f"Running {animal} {recording}") # NRK debugging add
        # diruse = os.path.join(BASE_DIR_ROOT, animal, recording)
        diruse = get_dir(animal, recording) # NRK compatibility add
        if not os.path.isdir(diruse):
            continue

        ripple_channel_id = chan_dict[animal][recording]
        sess = ProcessData(diruse)

        try:
            full_signal = sess.eegfile.get_signal()
        except Exception:
            print("Error in line 106") # NRK debug add
            continue

        recording_clean = recording.replace("_", "")
        injection_file = os.path.join(diruse, f"{animal_cap}_{recording_clean}_denoised.injection.npy")
        if not os.path.exists(injection_file):
            print("injection file missing") # NRK debug add
            continue

        inj_obj = np.load(injection_file, allow_pickle=True).item()
        post_idx = next((k for k, v in inj_obj['epochs']['label'].items() if v == "POST"), None)
        if post_idx is None:
            print("No POST epochs detected.")
            continue
        inj_time = float(inj_obj['epochs']['start'][post_idx])
        print(f"Injection time = {inj_time}")

        # Bug here - combined with lines below this was choosing the wrong channel to calculate peak frequency and power.
        # Importantly the correct channel was being used for initial ripple detection though...
        # chan_ids = np.arange(32, 0, -1)
        # shank = Shank().auto_generate(columns=1, contacts_per_column=32, channel_id=chan_ids)

        # NRK bugfix for the above is on the next three lines.
        chan_ids = np.arange(0, 32, 1)
        shank = Shank().auto_generate(columns=1, contacts_per_column=32, xpitch=0, ypitch=-50, channel_id=chan_ids)
        shank.set_disconnected_channels(sess.recinfo.skipped_channels) # NRK be sure to keep this in.
        sess.prbgrp = ProbeGroup()
        sess.prbgrp.add_probe(Probe(shank))

        
        art_epochs = load_artifact_epochs(diruse, animal_cap, recording_clean)
        ripple_epochs, _ = oscillations.detect_ripple_epochs(
            full_signal, sess.prbgrp,
            thresh=(FIXED_THRESHOLD, None),
            ripple_channel=ripple_channel_id,
            ignore_epochs=art_epochs,
            mindur=0.05,
            return_power=True
        )
        if len(ripple_epochs) == 0:
            continue

        # NRK bugfix: these lines were selecting the wrong channel for calculating ripple frequency and power.
        # The bugfix above on lines 133-135 also fixes this issue. However, the line below under the 'try' statement
        # is not necessary so I have commented it out. Setting the rpl_idx directly as the ripple_channel_id ensures that we are
        # always selecting the correct channel.
        # try:
        #    rpl_idx_in_signal = np.where(sess.prbgrp.channel_id == ripple_channel_id)[0][0]
        # except Exception:
        #    print("Error selecting rpl_idx_in_signal")
        rpl_idx_in_signal = ripple_channel_id  
        print([f"rpl_idx={rpl_idx_in_signal}, should match ripple channel {ripple_channel_id}!"])  # NRK debug add

        ripple_res = Ripple.get_peak_ripple_freq(
            sess.eegfile,
            rpl_channel=rpl_idx_in_signal,
            rpl_epochs=ripple_epochs
        )
        rpl_df = ripple_res.to_dataframe()

     
        if 'peak_time' not in rpl_df.columns:
            if 'peak' in rpl_df.columns:
                rpl_df = rpl_df.rename(columns={'peak': 'peak_time'})
            if 'peak_time' not in rpl_df.columns:
                rpl_df['peak_time'] = (rpl_df['start'] + rpl_df['stop']) / 2

        # 1 hour post-injection (0..3600) (relative to inj time)
        rpl_df['start']     = rpl_df['start']     - inj_time
        rpl_df['stop']      = rpl_df['stop']      - inj_time
        rpl_df['peak_time'] = rpl_df['peak_time'] - inj_time
        rpl_df = rpl_df[(rpl_df['peak_time'] >= 0) & (rpl_df['peak_time'] <= 3600)].copy()
        if rpl_df.empty:
            continue

        
        valid_epochs_df = pd.DataFrame({
            'start': rpl_df['start'] + inj_time,
            'stop':  rpl_df['stop']  + inj_time,
            'label': 'ripple'
        })
        valid_epochs = Epoch(valid_epochs_df)

        rpl_traces, t_frames = sess.eegfile.get_frames_within_epochs(
            valid_epochs,
            np.arange(sess.eegfile.n_channels),
            ret_time=True
        )

        rpl_traces = filter_sig.bandpass(rpl_traces, lf=2, hf=30, fs=sess.eegfile.sampling_rate)

        get_extrema = lambda arr: arr[np.argmax(np.abs(arr))]

        def process_swa(arr):
            # valid_epochs.flatten(): [start1, stop1, start2, stop2, ...]
            # statistic: get the max(|x|) value for each epoch
            out = stats.binned_statistic(t_frames, arr, bins=valid_epochs.flatten(), statistic=get_extrema)[0]
            return out[::2]  # those corresponding to epoch starts from start-stop pairs (your previous logic)

        max_val = Parallel(n_jobs=8)(delayed(process_swa)(arr) for arr in rpl_traces)

        rpl_df["sharp_wave_amplitude"] = np.ptp(np.asarray(max_val), axis=0) * 0.95 * 1e-3

        # labels
        rpl_df["session"] = recording_labels[recording]
        rpl_df["animal"] = animal
        group_ripple_data.append(rpl_df)

        print("\n") # NRK debug add to separate visually between each session's output to the console


# VISUALIZATION (single panel / feature)

if group_ripple_data:
    full_df = pd.concat(group_ripple_data, ignore_index=True)

    # Keep the animal order fixed
    animal_order = animals[:]  # ["rey","finn","rose","finn2"]
    animal_map = {a: f"Animal {i+1}" for i, a in enumerate(animal_order)}
    full_df["animal_label"] = full_df["animal"].map(animal_map)

    # NRK compatibility adds
    basedir_use = get_animal_dir("finn").parent
    full_df.to_csv(basedir_use / "ripple_features_group_from_final_plots.csv")

    final_pdf = os.path.join(basedir_use, "Manuscript_Final_Plots.pdf")

    features = {
        "duration": "Duration (s)",
        "peak_power": "Peak Power",
        "sharp_wave_amplitude": "SW Amplitude (mV)",
        "peak_frequency_bp": "Peak Frequency (Hz)"
    }


    pairs_sessions = [("Saline 1", "Psilocybin"),
                      ("Psilocybin", "Saline 2"),
                      ("Saline 1", "Saline 2")]

    with PdfPages(final_pdf) as pdf:
        for feat, ylab in features.items():

            df_plot = full_df.dropna(subset=[feat, "animal_label", "session"]).copy()

            fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2))

            
            sns.boxplot(
                data=df_plot,
                x="animal_label", y=feat,
                hue="session",
                order=[animal_map[a] for a in animal_order],
                hue_order=order_sessions,
                palette=palette,
                showfliers=False,
                fill=False,
                linewidth=0.8,
                ax=ax
            )

           
            sns.stripplot(
                data=df_plot,
                x="animal_label", y=feat,
                hue="session",
                order=[animal_map[a] for a in animal_order],
                hue_order=order_sessions,
                palette=palette,
                size=1.7,
                jitter=0.25,
                alpha=0.35,
                dodge=True,          
                ax=ax
            )

            
            handles, labels = ax.get_legend_handles_labels()
            
            if len(handles) >= 3:
                leg = ax.legend(handles[:3], ["S", "P", "S"], title="Session",
                                fontsize='x-small', title_fontsize='small',
                                frameon=True, loc="upper left")
            else:
                ax.legend_.remove() if ax.legend_ else None

        
            try:
                
                pairs = []
                for a_lbl in [animal_map[a] for a in animal_order]:
                    for (s1, s2) in pairs_sessions:
                        pairs.append(((a_lbl, s1), (a_lbl, s2)))

                annot = Annotator(
                    ax, pairs,
                    data=df_plot,
                    x="animal_label", y=feat,
                    hue="session",
                    order=[animal_map[a] for a in animal_order],
                    hue_order=order_sessions
                )
                annot.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=False)
                annot.apply_and_annotate()
            except Exception:
                pass

            ax.set_xlabel("")  
            ax.set_ylabel(ylab, fontweight='bold')
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            sns.despine(ax=ax)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=600)
            plt.close(fig)

    print(f"Successfully completed. PDF: {final_pdf}")
else:

    print("No data found: group_ripple_data is empty.")

