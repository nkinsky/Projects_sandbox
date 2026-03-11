function [inverted_data] = inv_bz_NormToRange(normdata, rawdata, range)
% [converted_values] = inv_bz_NormToRange(norm_values, raw_values)
%   inverse of bz_NormToRange. Takes previously normalized values and
%   converts them back to the raw_values data range.

if nargin < 3
    range = [0, 1]; % default for ClusterStates_GetMetrics
end

databounds(1) = min(rawdata(:)); databounds(2) = max(rawdata(:));
dataspan = diff(databounds);
rangespan = diff(range);

inverted_data = (normdata - range(1))./rangespan.*dataspan + databounds(1);

end