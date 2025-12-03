function [swthresh, EMGthreh, THthresh] = apply_state_thresh_across_sessions(basename_new_session, basename_good_session)
% [swthresh, EMGthreh, THthresh] = apply_state_thresh_across_sessions(basename_new_session, basename_good_session)
%   Determine the appropriate thresholds to apply for "new_session" based
%   on values obtained in "good_session" for determining sleep states.
% 
%   Useful for setting realistic thresholds from a "good_session" with both
%   sleep and wake for a "new_session" that has predominantly (or only) one
%   type of state.

%% First run good_session through ClusterStates_GetMetrics without normalizing EMG, SW, and theta arrays

EMGgood = importdata([basename_good_session, '.EMGFromLFP.LFP.mat']);
LFPgood = importdata([basename_good_session, '.SleepScoreLFP.LFP.mat']);

[basePath_good, ~, ~] = fileparts(basename_good_session);
[SleepScoreMetrics_good,~] = ClusterStates_GetMetrics(...
    basePath_good,LFPgood,EMGgood,false,'norm', false);

swthresh_good = SleepScoreMetrics_good.histsandthreshs.swthresh;
THthresh_good = SleepScoreMetrics_good.histsandthreshs.THthresh;
EMGthresh_good = SleepScoreMetrics_good.histsandthreshs.EMGthresh;

%% Next run new_session through ClusterStates_GetMetrics and get data ranges
EMGnew = importdata([basename_new_session, '.EMGFromLFP.LFP.mat']);
LFPnew = importdata([basename_new_session, '.SleepScoreLFP.LFP.mat']);

[basePath_new, ~, ~] = fileparts(basename_new_session);
[SleepScoreMetrics_new,~] = ClusterStates_GetMetrics(...
    basePath_new,LFPnew,EMGnew,false,'norm', false);

% Get data ranges
swdatarange = [min(SleepScoreMetrics_new.broadbandSlowWave), 
    max(SleepScoreMetrics_new.broadbandSlowWave)];
thdatarange = [min(SleepScoreMetrics_new.thratio), 
    max(SleepScoreMetrics_new.thratio)];
EMGdatarange = [min(SleepScoreMetrics_new.EMG), 
    max(SleepScoreMetrics_new.EMG)];

%% Third match up good thresholds to new data range after normalizing
swthresh_new_norm = bz_NormToRange(swthresh_good, [0, 1], swdatarange);
ththresh_new_norm = bz_NormToRange(THthresh_good, [0, 1], thdatarange);
EMGthresh_new_norm = bz_NormToRange(EMGthresh_good, [0, 1], EMGdatarange);

%% Finally plot everything
figure; subplot(2, 3, 1);
bar(SleepScoreMetrics_good.histsandthreshs.swhistbins, SleepScoreMetrics_good.histsandthreshs.swhist);
xline(swthresh_good, 'r');
title('SW raw good')

subplot(2, 3, 4);
bin_size = mean(diff(SleepScoreMetrics_good.histsandthreshs.swhistbins));
bins_use = SleepScoreMetrics_good.histsandthreshs.swhistbins - bin_size / 2;
bins_use = [bins_use, bins_use(end) + bin_size];
histogram(SleepScoreMetrics_good.broadbandSlowWave, bins_use);
xline(swthresh_good, 'r');
title('SW raw new')

subplot(2, 3, 2);
bar(SleepScoreMetrics_good.histsandthreshs.THhistbins, SleepScoreMetrics_good.histsandthreshs.THhist);
xline(THthresh_good, 'r');
title('TH raw good')

subplot(2, 3, 5);
MOVtimes = (SleepScoreMetrics_new.broadbandSlowWave(:)<swthresh_good & SleepScoreMetrics_new.EMG(:)>EMGthresh_good);
bin_size = mean(diff(SleepScoreMetrics_good.histsandthreshs.THhistbins));
bins_use = SleepScoreMetrics_good.histsandthreshs.THhistbins - bin_size / 2;
bins_use = [bins_use, bins_use(end) + bin_size];
histogram(SleepScoreMetrics_good.thratio(MOVtimes == 0), bins_use);
xline(THthresh_good, 'r');
title('TH raw new')

subplot(2, 3, 3);
bar(SleepScoreMetrics_good.histsandthreshs.EMGhistbins, SleepScoreMetrics_good.histsandthreshs.EMGhist);
xline(EMGthresh_good, 'r');
title('EMG raw good')

subplot(2, 3, 6);
bin_size = mean(diff(SleepScoreMetrics_good.histsandthreshs.EMGhistbins));
bins_use = SleepScoreMetrics_good.histsandthreshs.EMGhistbins - bin_size / 2;
bins_use = [bins_use, bins_use(end) + bin_size];
histogram(SleepScoreMetrics_good.EMG, bins_use);
xline(EMGthresh_good, 'r');
title('EMG raw new')

% NRK add in sanity check plotting of thresholds after normalizing!

end