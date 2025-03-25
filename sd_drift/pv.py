import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def calc_PV_corr(pf_df_in, filter_std=0, method: str in ["2D", "Rubin", "Mankin"] = "2D"):
    """Calculates population vectors Input is a nneurons x (nspatialbins * 2) dataframe
    with the last two columns denoting the group and recording session.
    Returns PVcorr, which gives correlations between all spatial bins in MAZE and reMAZE sessions.
    Also returns PVcorr_maze_remaze, consistent with Rubin et al. (2015) but different from Mankin et al. (2012)"""
    # Make into numpy array
    pf_array = pf_df_in.drop(columns=["grp", "session"]).to_numpy()
    assert (nbins_per_sesh := int(pf_array.shape[1] / 2)) == pf_array.shape[1] / 2, "odd number of spatial bins found, not sure how to divide in half"

    # Remove any rows with all zeroes or nans - Dont do this - all neurons included are stable, so if they aren't firing much during
    # running they legimately aren't active then but are active at other (non-running) times.
    # pf_array = pf_array[~np.all(pf_array == 0, axis=1)]
    # pf_array = pf_array[~np.all(np.isnan(pf_array), axis=1)]  # There shouldn't by any like this

    assert method in ["2D", "Rubin", "Mankin"]
    if method == "Rubin":
        # Calculate MAZE-reMAZE PV corr directly - gives ONE value. Consistent with Rubin et al. (2015)
        PVmaze = pf_array[:, 0:nbins_per_sesh]
        PVreMAZE = pf_array[:, nbins_per_sesh:]
        PVcorr_maze_remaze = np.corrcoef(PVmaze.reshape(-1), PVreMAZE.reshape(-1))[0, 1]
        PVcorr = PVcorr_maze_remaze

    elif method == "2D":

        # Remove any columns (spatial bins) which have all zeros in either MAZE or reMAZE
        bad_cols = np.where(np.all(pf_array == 0, axis=0))[0]
        bad_cols = np.hstack((bad_cols, bad_cols[bad_cols < nbins_per_sesh] + nbins_per_sesh))
        bad_cols = np.hstack((bad_cols, bad_cols[bad_cols >= nbins_per_sesh] - nbins_per_sesh))
        pf_array = np.delete(pf_array, np.unique(bad_cols), axis=1)

        # Calc PV corrs across all spatial bins separately and filter
        PVcorr = np.corrcoef(pf_array.T)
        if filter_std > 0:
            PVcorr = gaussian_filter(PVcorr, filter_std)

        # Reinsert bins not calucated as nans to keep everything the same shape
        for bad_col in np.unique(bad_cols):
            PVcorr = np.insert(PVcorr, bad_col, np.nan, axis=0)
            PVcorr = np.insert(PVcorr, bad_col, np.nan, axis=1)

    elif method == "Mankin":
        # assert False, "Mankin method not yet implemented"
        # Calc PV corrs across all spatial bins separately and filter
        PVcorr = np.corrcoef(pf_array.T)
        if filter_std > 0:
            PVcorr = gaussian_filter(PVcorr, filter_std)
        reMAZEcorr = PVcorr[0:nbins_per_sesh, nbins_per_sesh:(nbins_per_sesh * 2)]
        PVcorr_maze_remaze = np.diag(reMAZEcorr)
        PVcorr = PVcorr_maze_remaze

    return PVcorr


def calc_mean_PV_corr(pf_df_in, filter_std=0, noffset=3):
    """Calculates mean PV correlations for MAZE, reMAZE, and MAZE-reMAZE on the diagonal and off the diagonal.
    Input is an nneurons x (nspatialbins * 2) dataframe with the last two columns denoting the group and recording session.
    The first nspatialbins contain the mean firing rate for each neuron in that direction during MAZE,
    while the second nspatialbins are the same but for reMAZE"""

    # # Make into numpy array
    # pf_array = pf_df_in.drop(columns=["grp", "session"]).to_numpy()
    # assert (nbins_per_sesh := int(pf_array.shape[1]/2)) == pf_array.shape[1]/2, "odd number of spatial bins found, not sure how to divide in half"

    # # Remove any rows with all zeroes or nans - Dont do this - all neurons included are stable, so if they aren't firing much during
    # # running they legimately aren't active then but are active at other (non-running) times.
    # # pf_array = pf_array[~np.all(pf_array == 0, axis=1)]
    # # pf_array = pf_array[~np.all(np.isnan(pf_array), axis=1)]  # There shouldn't by any like this

    # # Also calculate MAZE-reMAZE PV corr directly - gives ONE value. Consistent with Rubin et al. (2015)
    # PVmaze = pf_array[:, 0:nbins_per_sesh]
    # PVreMAZE = pf_array[:, nbins_per_sesh:]
    # PVcorr_maze_remaze = np.corrcoef(PVmaze.reshape(-1), PVreMAZE.reshape(-1))[0, 1]

    # # Note that Mankin does it a bit differently, calculating the PV corr at each spatial bin and then averaging
    # # or calculating the mean PV across all bins

    # # Remove any columns (spatial bins) which have all zeros in either MAZE or reMAZE
    # bad_cols = np.where(np.all(pf_array == 0, axis=0))[0]
    # bad_cols = np.hstack((bad_cols, bad_cols[bad_cols < nbins_per_sesh] + nbins_per_sesh))
    # bad_cols = np.hstack((bad_cols, bad_cols[bad_cols >= nbins_per_sesh] - nbins_per_sesh))
    # pf_array = np.delete(pf_array, np.unique(bad_cols), axis=1)

    # # Calc PV corrs across all spatial bins separately and filter
    # PVcorr = np.corrcoef(pf_array.T)
    # if filter_std > 0:
    #     PVcorr = gaussian_filter(PVcorr, filter_std)

    # Calculate PV corrs at all time bins
    PVcorr = calc_PV_corr(pf_df_in, filter_std=filter_std, method="2D")
    PVcorr_maze_remaze_Rubin = calc_PV_corr(pf_df_in, filter_std=filter_std, method="Rubin")
    PVcorr_maze_remaze_Mankin = np.mean(calc_PV_corr(pf_df_in, filter_std=filter_std, method="Mankin"))

    # Segregate PVcorr array into appropriate session comparisons
    assert (nbins_per_sesh := int(PVcorr.shape[1]/2)) == PVcorr.shape[1]/2, "odd number of spatial bins found after removing all-zero bins"
    maze_corr = PVcorr[0:nbins_per_sesh, 0:nbins_per_sesh]
    remaze_corr = PVcorr[nbins_per_sesh:(nbins_per_sesh * 2), nbins_per_sesh:(nbins_per_sesh * 2)]
    maze_remaze_corr = PVcorr[0:nbins_per_sesh, nbins_per_sesh:(nbins_per_sesh * 2)]

    # Calculate on-diagonal and off-diagonal mean corrs
    corr_all, fname = [], []
    for corr_name, corr_use in zip(["MAZE", "reMAZE", "MAZE_reMAZE"], [maze_corr, remaze_corr, maze_remaze_corr]):
        c_diag = np.tril(np.triu(corr_use, -noffset), noffset)
        c_off = np.triu(corr_use, noffset + 1) + np.tril(corr_use, -noffset - 1)

        # Calculate mean value on diagonal
        corr_all.append(np.nanmean(c_diag[c_diag != 0]))
        fname.append(f"{corr_name}_diag")

        # Calculate mean value on off-diagonal
        corr_all.append(np.nanmean(c_off[c_off != 0]))
        fname.append(f"{corr_name}_off")

    # Make into one dataframe
    pv_sesh = pd.DataFrame({f: corr for f, corr in zip(fname, corr_all)}, index=[0])
    pv_sesh["PVcorr_maze_remaze_Rubin"] = PVcorr_maze_remaze_Rubin
    pv_sesh["PVcorr_maze_remaze_Mankin"] = PVcorr_maze_remaze_Mankin
    pv_sesh["grp"] = pf_df_in.reset_index().loc[0, "grp"]
    # pv_sesh["session"] = pf_df_in.loc[0, "session"]

    return pv_sesh