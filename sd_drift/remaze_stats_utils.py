import numpy as np
from scipy import stats
import pandas as pd
from joblib import Parallel, delayed
import functools
from tqdm import tqdm
from copy import copy


def resample(df, level=['session_type', 'sid', 'mean_frate'], n_level=None, apply=None):
    """Resample data with replacement at each level indicated.
    n_level = number of samples to grab at each level. If None, it will resample with replacement from each level n times,
    where n is the number of unique values in that level. If not None, it must be a list the same length as `level`
    with either None or an int for the number of samples to grab from the corresponding level.
    e.g., if you wanted to resample mean firing rates for MAZE vs REMAZE from all rats but only grab ONE session from each rat
    for each bootstrap, you would enter:
    >>> resample(df, level=['session_type', 'sid', 'mean_frate'], n_level=[None, 1, None])"""

    if apply is not None:
        assert callable(apply), "apply can only be a function"
        df = resample(df, level=level, n_level=n_level, apply=None)
        new_df = apply(df)

    else:
        n_level = [None]*len(level) if n_level == None else n_level
        assert (len(n_level) == len(level)) & np.all([isinstance(_, int) or _ is None for _ in n_level]), "`n_level` must be a list of ints and None the same length as `level`"
        if len(level) > 1:
            param = level[0]  # Get name of level to resample at
            next_levels = copy(level[1:])  # Get next levels to resample
            next_n_levels = copy(n_level[1:])
            ids = df[param].unique()  # Grab unique values to resample from, e.g. animal names or session ids
            n_samples = len(ids) if n_level[0] is None else n_level[0]

            # Now resample
            rng = np.random.default_rng()
            # resample_ids = rng.choice(ids, size=len(ids), replace=True)
            resample_ids = rng.choice(ids, size=n_samples, replace=True)

            new_df = []
            # Loop through and generate a new dataframe for each id in the resample ids
            for i, idx in enumerate(resample_ids):
                idx_df = df[df[param] == idx].copy()
                idx_df.loc[:, param] = i  # Make each sample "independent"

                # Recursively call resample to resample at the next level(s)
                # idx_df = resample(idx_df, level=next_levels, apply=apply)
                idx_df = resample(idx_df, level=next_levels, n_level=next_n_levels, apply=apply)
                new_df.append(idx_df)

            new_df = pd.concat(new_df, ignore_index=True)

            # if apply is not None:
            #     assert callable(apply), "apply can only be a function"
            #     new_df = apply(new_df)

        elif len(level) == 1:  # If at the bottom level, actually resample!
            assert level[0] in df.keys(), 'Last parameter in "level" not in keys of df'
            # new_df = [df.sample(frac=1, replace=True, ignore_index=True)]
            new_df = df.sample(frac=1, replace=True, ignore_index=True)

    return new_df


def bootstrap_resample(df: pd.DataFrame, n_iter, n_jobs=1, apply=None,
                       level=['session_type', 'sid', 'mean_frate'], n_level=None):
    # groups = df["grp"].unique()

    partial_resample = functools.partial(resample, level=level, n_level=n_level, apply=apply)
    out_df = []
    # for grp in groups:
    #     print(f"Running bootstraps for {grp} group")
    #     df_grp = df[df["grp"] == grp]
    data = [
        r
        for r in tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(partial_resample)(df) for _ in range(n_iter)
            ),
            total=n_iter,
            # position=1,
            # leave=False,
        )
    ]
    data = pd.concat(data, ignore_index=True)
        # data["grp"] = grp
        # out_df.append(data)

    # return pd.concat(out_df, ignore_index=True)
    return data
