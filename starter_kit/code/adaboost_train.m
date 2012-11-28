function [ boost ] = adaboost_train(Xt_raw, Yt, T)%, predictors, T)
% Definitely not working right now.

%numsplits = 20;

%Ycounts = zeros(10, 1);
%for i = 1:10
%    Ycounts(i) = sum(Yt==i);
%end

%predictors = cell(numsplits*size(Xt_raw, 2),1);
%for i = 1:size(Xt_raw, 2)
%    min_x = min(Xt_raw(:,i));
%    max_x = max(Xt_raw(:,i));
%    for j = 1:numsplits
%        threshold = min_x+(j-1)*(max_x-min_x)/4;
%        Y_threshold = Yt(Xt_raw(:,i)<threshold);
%        Y_threshold_counts = zeros(10, 1);
%        for k = 1:10
%            Y_threshold_counts(k) = sum(Y_threshold==k);
%        end
%        Y_threshold_counts = Y_threshold_counts./Ycounts;
%        [~, under] = max(Y_threshold_counts);
%        [~, over] = min(Y_threshold_counts);
%        predictors{numsplits*(i-1)+j} = @(x) decision_stump (x, threshold, under, over);
%    end
%end

load 'forests_model_d10.mat'

%try 1-NN on each feature
%predictors = cell(size(Xt_raw, 2), 1);
%for i = 1:numel(predictors)
%    predictors{i} = @(xt,xq, y) knn_by_feature(xt, xq, y, i, 50);
%end

N = numel(Yt);
M = numel(model.audio_forest);

expectations = zeros(N,M);

for i = 1:M
    for j = 1:N
        yhat = dt_value(model.audio_forest(i), Xt_raw(j,:));
        expectations(j, i) = yhat;
    end
    %f = predictors{i};
    %[~, idx] = max(f(Xt_raw, Xt_raw, Yt), [], 2);
    %expectations(:,i) = idx;
end

errors = repmat(Yt, 1, M) ~= expectations;
y_times_h = zeros(size(errors));
y_times_h(errors==1) = -1;
y_times_h(errors==0) = 1;

D = ones(size(Yt));
D = D/N;

h = zeros(T, 1);
alpha = zeros(T, 1);
%feature_index = zeros(T, 1);
for t = 1:T
    err_array = sum(repmat(D, 1, M).*(1/2 - (1-errors)/2 + errors/18), 1);
    [epsilon, i] = min(err_array);
    h(t) = i;
    %feature_index(t) = ceil(i);
    alpha(t) = log((1 - epsilon)/epsilon)/2;
    Z = sum(D.*exp(-alpha(t)*y_times_h(:, i)), 1);
    D = D.*exp(-alpha(t)*y_times_h(:, i))/Z;
end

boost.h = h;
boost.alpha = alpha;
%boost.feature_index = feature_index;