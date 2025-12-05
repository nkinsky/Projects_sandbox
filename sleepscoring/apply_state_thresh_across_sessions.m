function [swthresh_new_norm, EMGthresh_new_norm, THthresh_new_norm] = apply_state_thresh_across_sessions( ...
    basename_new_session, basename_good_session, exclude_nrem_times)
% [swthresh, EMGthreh, THthresh] = apply_state_thresh_across_sessions(basename_new_session, basename_good_session)
%   Determine the appropriate thresholds to apply for "new_session" based
%   on values obtained in "good_session" for determining sleep states.
% 
%   Useful for setting realistic thresholds from a "good_session" with both
%   sleep and wake for a "new_session" that has predominantly (or only) one
%   type of state.
%
%  set 3rd argument ('exclude_nrem_times') to true if you see a warning pop
%  up that says "No bimodal dip found in theta. Trying to exclude NREM..."
%  while running.

% don't exclude nrem times by default
if nargin < 3
    exclude_nrem_times = false;
end

%% First run good_session through ClusterStates_GetMetrics without normalizing EMG, SW, and theta arrays

EMGgood = importdata([basename_good_session, '.EMGFromLFP.LFP.mat']);
LFPgood = importdata([basename_good_session, '.SleepScoreLFP.LFP.mat']);

disp('Running good session')
[basePath_good, ~, ~] = fileparts(basename_good_session);
[SleepScoreMetrics_good,~] = ClusterStates_GetMetrics(...
    basePath_good,LFPgood,EMGgood,false,'norm', false);

% Get thresholds
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

% Get good and new EMG thresholds


% Get data ranges
swdatarange = [min(SleepScoreMetrics_new.broadbandSlowWave), 
    max(SleepScoreMetrics_new.broadbandSlowWave)];

if strcmp(emg_or_motion, 'emg')
    EMGthresh_new_orig = SleepScoreMetrics_new.histsandthreshs.EMGthresh;
    EMGdatarange = [min(SleepScoreMetrics_new.EMG), 
        max(SleepScoreMetrics_new.EMG)];
    EMGhistbins = SleepScoreMetrics_good.histsandthreshs.EMGhistbins;
    EMGhistbins_orig = SleepScoreMetrics_new.histsandthreshs.EMGhistbins;
    EMGhist = SleepScoreMetrics_good.histsandthreshs.EMGhist;
    EMGgood = SleepScoreMetrics_good.EMG;
    EMGnew_use = SleepScoreMetrics_new.EMG;
elseif strcmp(emg_or_motion, 'motion')
    EMGthresh_new_orig = SleepScoreMetrics_new.histsandthreshs.MotionThresh;
    EMGdatarange = [min(SleepScoreMetrics_new.motiondata), 
        max(SleepScoreMetrics_new.motiondata)];
    EMGhistbins = SleepScoreMetrics_good.histsandthreshs.MotionHistBins;
    EMGhistbins_orig = SleepScoreMetrics_new.histsandthreshs.MotionHistBins;
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
hg = bar(SleepScoreMetrics_good.histsandthreshs.swhistbins, ...
    SleepScoreMetrics_good.histsandthreshs.swhist / sum(SleepScoreMetrics_good.histsandthreshs.swhist));
xlabel('Broadband SW')
ylabel('Probability')
title('Both sessions: Raw values')

for j = 1:2
    norm_type = 'probability';
    if j == 1; hold on; else; subplot(3, 3, 4); norm_type = 'count'; end
    bin_size = mean(diff(SleepScoreMetrics_good.histsandthreshs.swhistbins));
    bins_use = SleepScoreMetrics_good.histsandthreshs.swhistbins - bin_size / 2;
    bins_use = [bins_use, bins_use(end) + bin_size];
    hn = histogram(SleepScoreMetrics_new.broadbandSlowWave, bins_use, 'Normalization', norm_type);
    ht = xline(swthresh_good, 'r');
    if j == 1
        legend([hg, hn, ht], ["Good", "New", "Raw thresh"]);
    else 
        title('New session only: Raw values'); 
        xlabel('Broadband SW'); 
        ylabel('Count');
    end
end
% title('SW raw new')

subplot(3, 3, 2);
hg = bar(SleepScoreMetrics_good.histsandthreshs.THhistbins, ...
    SleepScoreMetrics_good.histsandthreshs.THhist / sum(SleepScoreMetrics_good.histsandthreshs.THhist));
xlabel('Theta / all ratio')
ylabel('Probability')
title('Both sessions: Raw values')

for j = 1:2
    norm_type = 'probability';
    if j == 1; hold on; else; subplot(3, 3, 5); norm_type = 'count'; end

    bin_size = mean(diff(SleepScoreMetrics_good.histsandthreshs.THhistbins));
    bins_use = SleepScoreMetrics_good.histsandthreshs.THhistbins - bin_size / 2;
    bins_use = [bins_use, bins_use(end) + bin_size];
    hn = histogram(thratio_new, bins_use, 'Normalization', norm_type);
    ht = xline(THthresh_good, 'r');
    if j == 1
        legend([hg, hn, ht], ["Good", "New", "Raw thresh"]);
    else 
        title('New session only: Raw values'); 
        xlabel('Theta / all ratio'); 
        ylabel('Count');
    end
end

subplot(3, 3, 3);
hg = bar(EMGhistbins, EMGhist / sum(EMGhist));
xline(EMGthresh_good, 'r');
xlabel('EMG/Motion')
ylabel('Probability')
title('Both sessions: Raw values')

for j = 1:2
    norm_type = 'probability';
    if j == 1; hold on; else; subplot(3, 3, 6); norm_type = 'count'; end
    bin_size = mean(diff(EMGhistbins));
    bins_use = EMGhistbins - bin_size / 2;
    bins_use = [bins_use, bins_use(end) + bin_size];
    hn = histogram(EMGnew_use, bins_use, 'Normalization', norm_type);
    ht = xline(EMGthresh_good, 'r');
    if j == 1
        legend([hg, hn, ht], ["Good", "New", "Raw thresh"]);
    else 
        title('New session only: Raw values'); 
        xlabel('EMG/Motion'); 
        ylabel('Count');
    end
end

% Sanity check plotting of data and thresholds after normalizing!
subplot(3, 3, 7);
swthresh_orig_norm = bz_NormToRange(SleepScoreMetrics_new.histsandthreshs.swthresh, ...
    [0, 1], swdatarange);
histogram(bz_NormToRange(SleepScoreMetrics_new.broadbandSlowWave, [0, 1]), ...
    bz_NormToRange(SleepScoreMetrics_new.histsandthreshs.swhistbins, [0, 1], swdatarange))
hn = xline(swthresh_new_norm, 'r');
horig = xline(swthresh_orig_norm, 'g--');
title('New session: normalized values')
xlabel('Broadband SW')
ylabel('Count')
legend([hn, horig], ["New thresh", "Orig thresh"]);

subplot(3, 3, 8);
MOVtimes_orig = ((SleepScoreMetrics_new.broadbandSlowWave(:)<SleepScoreMetrics_new.histsandthreshs.swthresh) ...
    & (EMGnew_use(:) > EMGthresh_new_orig));
thratio_new_orig = SleepScoreMetrics_new.thratio(MOVtimes_orig == 0);
% THthresh_orig_norm = bz_NormToRange(SleepScoreMetrics_new.histsandthreshs.THthresh, ...
%     [0,1], SleepScoreMetrics_new.histsandthreshs.THhistbins([1, end]));
THthresh_orig_norm = bz_NormToRange(SleepScoreMetrics_new.histsandthreshs.THthresh, ...
    [0,1], [min(thratio_new_orig), max(thratio_new_orig)]);
histogram(bz_NormToRange(thratio_new_orig, [0, 1]), ...
    bz_NormToRange(SleepScoreMetrics_new.histsandthreshs.THhistbins, [0, 1], thdatarange));
hn = xline(THthresh_new_norm, 'r');
horig = xline(THthresh_orig_norm, 'g--');
xlabel('TH / all ratio')
ylabel('Count')
title('New session: normalized values')
legend([hn, horig], ["New thresh", "Orig thresh"]);

subplot(3, 3, 9);
EMGthresh_orig_norm = bz_NormToRange(EMGthresh_new_orig, [0, 1], EMGdatarange);
histogram(bz_NormToRange(EMGnew_use, [0, 1]), ...
    bz_NormToRange(EMGhistbins_orig, [0, 1], EMGdatarange));
hn = xline(EMGthresh_good, 'r');
horig = xline(EMGthresh_orig_norm, 'g--');
xlabel('EMG / Motion')
ylabel('Count')
title('New session: normalized values')
legend([hn, horig], ["New thresh", "Orig thresh"]);


end