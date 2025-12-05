function [swthresh_new_norm, EMGthresh_new_norm, THthresh_new_norm] = apply_state_thresh_across_sessions(basename_new_session, basename_good_session)
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

disp('Running good session')
[basePath_good, ~, ~] = fileparts(basename_good_session);
[SleepScoreMetrics_good,~] = ClusterStates_GetMetrics(...
    basePath_good,LFPgood,EMGgood,false,'norm', false);

swthresh_good = SleepScoreMetrics_good.histsandthreshs.swthresh;
THthresh_good = SleepScoreMetrics_good.histsandthreshs.THthresh;
try
    EMGthresh_good = SleepScoreMetrics_good.histsandthreshs.EMGthresh;
    emg_or_motion = 'emg';
catch
    EMGthresh_good = SleepScoreMetrics_good.histsandthreshs.MotionThresh; 
    emg_or_motion = 'motion';
end


%% Next run new_session through ClusterStates_GetMetrics and get data ranges
EMGnew = importdata([basename_new_session, '.EMGFromLFP.LFP.mat']);
LFPnew = importdata([basename_new_session, '.SleepScoreLFP.LFP.mat']);

disp('Running new session')
[basePath_new, ~, ~] = fileparts(basename_new_session);
[SleepScoreMetrics_new,~] = ClusterStates_GetMetrics(...
    basePath_new,LFPnew,EMGnew,false,'norm', false);

% Get data ranges
swdatarange = [min(SleepScoreMetrics_new.broadbandSlowWave), 
    max(SleepScoreMetrics_new.broadbandSlowWave)];

if strcmp(emg_or_motion, 'emg')
    EMGdatarange = [min(SleepScoreMetrics_new.EMG), 
        max(SleepScoreMetrics_new.EMG)];
    EMGhistbins = SleepScoreMetrics_good.histsandthreshs.EMGhistbins;
    EMGhist = SleepScoreMetrics_good.histsandthreshs.EMGhist;
    EMGgood = SleepScoreMetrics_good.EMG;
    EMGnew_use = SleepScoreMetrics_new.EMG;
elseif strcmp(emg_or_motion, 'motion')
    EMGdatarange = [min(SleepScoreMetrics_new.motiondata), 
        max(SleepScoreMetrics_new.motiondata)];
    EMGhistbins = SleepScoreMetrics_good.histsandthreshs.MotionHistBins;
    EMGhist = SleepScoreMetrics_good.histsandthreshs.MotionHist;
    EMGgood = SleepScoreMetrics_good.motiondata;
    EMGnew_use = SleepScoreMetrics_new.motiondata;

MOVtimes = (SleepScoreMetrics_new.broadbandSlowWave(:)<swthresh_good & EMGnew_use(:)>EMGthresh_good);
thratio_new = SleepScoreMetrics_new.thratio(MOVtimes == 0);
thdatarange = [min(thratio_new), max(thratio_new)];

%% Third match up good thresholds to new data range after normalizing
swthresh_new_norm = bz_NormToRange(swthresh_good, [0, 1], swdatarange);
if all(MOVtimes)
    THthresh_new_norm = nan;
    disp("No immobile times detected in EMG/motion. Should be no REM, set theta_thresh = inf.")
else
    THthresh_new_norm = bz_NormToRange(THthresh_good, [0, 1], thdatarange);
end
EMGthresh_new_norm = bz_NormToRange(EMGthresh_good, [0, 1], EMGdatarange);

disp(['Use swthresh=' num2str(swthresh_new_norm) 'for new session'])
disp(['Use THthresh=' num2str(THthresh_new_norm) 'for new session'])
disp(['Use EMGthresh=' num2str(EMGthresh_new_norm) 'for new session'])

%% Finally plot everything
figure; subplot(3, 3, 1);
bar(SleepScoreMetrics_good.histsandthreshs.swhistbins, SleepScoreMetrics_good.histsandthreshs.swhist);
xline(swthresh_good, 'r');
title('SW raw good')

subplot(3, 3, 4);
bin_size = mean(diff(SleepScoreMetrics_good.histsandthreshs.swhistbins));
bins_use = SleepScoreMetrics_good.histsandthreshs.swhistbins - bin_size / 2;
bins_use = [bins_use, bins_use(end) + bin_size];
histogram(SleepScoreMetrics_new.broadbandSlowWave, bins_use);
xline(swthresh_good, 'r');
title('SW raw new')

subplot(3, 3, 2);
bar(SleepScoreMetrics_good.histsandthreshs.THhistbins, SleepScoreMetrics_good.histsandthreshs.THhist);
xline(THthresh_good, 'r');
title('TH raw good')

subplot(3, 3, 5);
bin_size = mean(diff(SleepScoreMetrics_good.histsandthreshs.THhistbins));
bins_use = SleepScoreMetrics_good.histsandthreshs.THhistbins - bin_size / 2;
bins_use = [bins_use, bins_use(end) + bin_size];
histogram(thratio_new, bins_use);
xline(THthresh_good, 'r');
title('TH raw new')

subplot(3, 3, 3);
bar(EMGhistbins, EMGhist);
xline(EMGthresh_good, 'r');
title('EMG raw good')

subplot(3, 3, 6);
bin_size = mean(diff(EMGhistbins));
bins_use = EMGhistbins - bin_size / 2;
bins_use = [bins_use, bins_use(end) + bin_size];
histogram(EMGnew_use, bins_use);
xline(EMGthresh_good, 'r');
title('EMG raw new')

% NRK add in sanity check plotting of thresholds after normalizing!
sublot(3,3,7);
bins_use = SleepScoreMetrics_good.histsandthreshs.swhistbins;
histogram(SleepScoreMetrics_new.broadbandSlowWave, length(bins_use) + 1);
xline(swthresh_new_norm, 'r');
title('SW raw new')

subplot(3, 3, 8);
bins_use = SleepScoreMetrics_good.histsandthreshs.THhistbins;
histogram(thratio_new, length(bins_use) + 1);
xline(THthresh_new_norm, 'r')
title('TH raw new full')

subplot(3, 3, 9);
bins_use = EMGhistbins - bin_size / 2;
histogram(EMGnew, length(bins_use));
xline(EMGthresh_good, 'r');

end