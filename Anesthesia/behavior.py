import pandas as pd


def get_lick_data(file_loc):
    """Parses behavioral lick data from a linear track from Brian Kim generated txt file using 'WaterTraining_BK'
    in MATLAB

    :param: file_loc: location of *Peformance_YYYYMMDD.txt file"""

    behave_data = pd.read_csv(file_loc, delimiter="; ", engine="python")
    lick_loc = behave_data.iloc[:, 1].apply(lambda x: x.split(": ")[1]).astype(int)
    lick_time = behave_data.iloc[:, 0].apply(lambda x: x.split(" Trial: ")[0])
    lick_time = pd.to_datetime(lick_time, format="%d-%b-%Y %I:%M:%S:%f")
    trial_num = behave_data.iloc[:, 0].apply(lambda x: x.split(" Trial: ")[1]).astype(int)

    lick_df = pd.DataFrame({"Datetime": lick_time, "Trial": trial_num, "Port": lick_loc})
    lick_df.loc[1:, "Correct"] = (lick_df["Port"].diff().abs() == 1)
    lick_df.loc[0, "Correct"] = True
    lick_df["Correct"] = lick_df["Correct"].astype(int)

    return lick_df