function [] = sleepstate2csv(dir_use)
% sleepstate2csv(dir_use)
%   Converts sleepstates to CSV files.
if nargin == 0
    dir_use = pwd;
end

% Change to dir_use and find file
curr_dir = pwd;
cd(dir_use)
file_use = dir("*.SleepStateEpisodes.states.mat");

% Load in file
SleepStateEpisodes = importdata(file_use.name);

% Parse data
WAKEep = SleepStateEpisodes.ints.WAKEepisode;
WAKEep(:, 3) = 1;
NREMep = SleepStateEpisodes.ints.NREMepisode;
NREMep(:, 3) = 2;
REMep = SleepStateEpisodes.ints.REMepisode;
REMep(:, 3) = 3;
names = {"WAKE", "NREM", "REM"};

% Combine
comb = [WAKEep; NREMep; REMep];

%$ Sort by time
[~, isort] = sort(comb, 1); 
comb_sorted = comb(isort(:, 1), :);

% Convert to cell
comb_sort_cell = cell(size(comb_sorted));
comb_sort_cell(:,1:2) = num2cell(comb_sorted(:,1:2));
comb_sort_cell(:, 3) = names(comb_sorted(:,3) + 1);

% Save
writecell(comb_sort_cell, fullfile(pwd, "brainstates.csv"))
disp('Brainstates written to' + fullfile(pwd, "brainstates.csv"))

% Change back to old directory
cd(curr_dir)

end