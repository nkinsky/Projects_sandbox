import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from neuropy.io.neuroscopeio import NeuroscopeIO
from neuropy.io.binarysignalio import BinarysignalIO
from neuropy.core import Epoch
from Psilocybin.subjects import get_animal_num
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plot_dir = Path("/Users/nkinsky/Library/CloudStorage/Dropbox-UniversityofMichigan/Nathaniel Kinsky/Manuscripts/Psilocybin/plots")

# Channel dictionary from HVS_Detection.py
chan_dict = {
    "Finn":  {"Saline1": 27, "Psilocybin": 27, "Saline2": 27},
    "Rey":   {"Saline1": 21, "Psilocybin": 21, "Saline2": 21},
    "Rose":  {"Saline1": 26, "Psilocybin": 26, "Saline2": 26},
    "Finn2": {"Saline1": 4,  "Psilocybin": 4,  "Saline2": 4}
}

hvfile_dict = {"Rey": "hv3", "Finn": "hv3", "Rose": "hv3", "Finn2": "hv3"}

# Date dictionary - UPDATE THESE FOR EACH ANIMAL AS NEEDED
date_dict = {
    "Rey": {
        "Saline1": "2022_06_01",
        "Psilocybin": "2022_06_02",
        "Saline2": "2022_06_03"
    },
    "Finn": {
        "Saline1": "2022_02_15",
        "Psilocybin": "2022_02_17",
        "Saline2": "2022_02_18"
    },
    "Rose": {
        "Saline1": "2022_08_09",
        "Psilocybin": "2022_08_10",
        "Saline2": "2022_08_11"
    },
    "Finn2": {
        "Saline1": "2023_05_24",
        "Psilocybin": "2023_05_25",
        "Saline2": "2023_05_26"
    }
}

# List of animals and sessions
animals = ["Finn", "Rey", "Rose", "Finn2"]
sessions = ["Saline1", "Psilocybin", "Saline2"]

# Collect data
data = []
for animal in animals:
    for session in sessions:
        date = date_dict[animal][session]
        session_lower = session.lower()
        base_path = f"/Users/nkinsky/Documents/UM/Working/Psilocybin/Recording_Rats/{animal}/{date}_{session_lower}"

        print(f"Processing {animal} {session}: {base_path}")

        # Find HVS epochs file using glob
        base_dir = Path(base_path)
        hvs_files = list(base_dir.glob("*.hvs_epochs.npy"))
        if not hvs_files:
            print(f"No .hvs_epochs.npy file found in {base_dir}")
            continue
        hvs_file = sorted(hvs_files)[0]

        try:
            # Load HVS epochs
            hvs_epochs = Epoch(epochs=None, file=hvs_file)
            print(f"Loaded HVS epochs: {len(hvs_epochs)} epochs")

            # Load injection time
            inj_epochs = Epoch(epochs=None, file=sorted(Path(base_path).glob("*.injection.npy"))[0])
            inj_time = inj_epochs["POST"].starts[0]
            print(f"Injection time: {inj_time} seconds")

            # Zero HVS epochs to 0 = inj time
            hvs_epochs = hvs_epochs.shift(-inj_time)

            # Filter to first hour (0-3600 seconds)
            hvs_epochs = hvs_epochs.time_slice(0, np.min((3600, hvs_epochs.stops.max())))

            # Sum durations for entire session
            total_hvs_time = hvs_epochs.durations.sum()
            print(f"Total HVS time: {total_hvs_time}")

            # Get session duration from eeg file
            xml_file = sorted(base_dir.glob("*.xml"))[0]
            recinfo = NeuroscopeIO(xml_file)
            # Calculate duration from file size: size / (bytes_per_sample * n_channels) / sampling_rate
            file_size = recinfo.eeg_filename.stat().st_size
            bytes_per_sample = 2  # int16
            n_samples = file_size / (bytes_per_sample * recinfo.n_channels)
            duration = n_samples / recinfo.eeg_sampling_rate
            print(f"Session duration: {duration}")

            # Calculate proportion as percentage
            proportion = (total_hvs_time / duration) * 100
            print(f"Proportion: {proportion}%")

            # Append to data
            data.append({
                "animal": animal,
                "animal_num": get_animal_num(animal),
                "session": session_lower,
                "proportion": proportion
            })
        except Exception as e:
            print(f"Error processing {hvs_file}: {e}")
            continue

print(f"Collected data for {len(data)} entries")

# Create DataFrame
df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(df.head())

# Plot using seaborn stripplot
if df.empty:
    print("No data collected to plot")
else:
    plt.figure(figsize=(3, 2.5), layout='tight')
    # Define custom palette for more distinct colors
    custom_palette = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}
    sns.stripplot(data=df, x="session", y="proportion", hue="animal_num", dodge=True, palette=custom_palette)
    plt.title("Proportion of Time Spent in HVS")
    plt.xlabel("Session")
    plt.ylabel("Proportion of Time in HVS (%)")
    plt.legend(title="Animal Number")
    sns.despine(ax=plt.gca())
    plt.savefig(plot_dir / "HVSTotalPlot.pdf")
    plt.show()
