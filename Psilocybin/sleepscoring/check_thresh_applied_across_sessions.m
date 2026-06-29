function check_thresh_applied_across_sessions(basename_session)
% check_thresh_applied_across_sessions(basename_session)
%   Check if thresholds were applied properly. Output the threshold used in
%   raw data values, NOT min/max normalized values.

if nargin < 1
    [~, n, ~] = fileparts(pwd);
    basename_session = fullfile(pwd, n);
end

%% Get Rat name and session name
[f, ~, ~] = fileparts(basename_session);
[f2, n2, ~] = fileparts(f);
[~, n3, ~] = fileparts(f2);
rat = n3;
session = n2;

%% Import data

rawmetrics = importdata([basename_session, '.SleepScoreMetrics_raw.LFP.mat']);
SleepState = importdata([basename_session, '.SleepState.states.mat']);

%% Check value for each

metric_names = ["broadbandSlowWave_raw", "thratio_raw", "motiondata_raw"];
thresh_names = ["swthresh", "THthresh", "MotionThresh"];

disp(rat + " " + session)
for i=1:3
    % Get raw metric values, threshold used (in min/max normalized space),
    % and good times to use
    metric = rawmetrics.(metric_names(i));
    thresh = SleepState.detectorinfo.detectionparms.SleepScoreMetrics.histsandthreshs.(thresh_names(i));
    good_times = SleepState.detectorinfo.detectionparms.SleepScoreMetrics.t_clus;

    % Grab only good times (no artifacts)
    metric_good = metric(good_times);

    % Calculate raw data threshold
    data_ptp = max(metric_good) - min(metric_good);
    raw_thresh = min(metric_good) + data_ptp*thresh;

    disp(thresh_names(i) + " = " + raw_thresh)

end