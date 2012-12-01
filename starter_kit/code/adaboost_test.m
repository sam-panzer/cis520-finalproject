function [ yhat ] = adaboost_test(Xq_raw, boost)

load 'forests_model_d10.mat'

N = size(Xq_raw, 1);

T = numel(boost.h);

predictions = zeros(N, 10, T);
for t = 1:T
    h = boost.h(t);
    for j = 1:N
        predictions(j, :, t) = boost.alpha(t) * dt_value_ranked(model.audio_forest(h), Xq_raw(j,:));
    %[~, idx] = max(h(Xt_raw ,Xq_raw, Yt),[], 2);
    %predictions(:,t) = idx;
    end
end
    
yhat = sum(predictions,3);


%for i = 1:10
%    alpha = repmat(boost.alpha, N, 1);
%    vote = zeros(N, T);
%    vote(predictions==i) = alpha(predictions==i);
%    yhat(:,i) = sum(vote,2);
%end

end