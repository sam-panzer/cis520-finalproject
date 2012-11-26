function [ yhat ] = knn(Xt_raw, Xq_raw, Yt, K)
% knn Summary of this function goes here
%   Detailed explanation goes here
Xt = bsxfun(@rdivide, Xt_raw, sum(Xt_raw, 2));
Xq = bsxfun(@rdivide, Xq_raw, sum(Xq_raw, 2));

D = Xq*Xt';
[~, idx] = sort(D, 2, 'descend');
ynn = idx(:, 1:K);
yhat = zeros(size(Xq, 1), 10);
yvals = Yt(ynn);
for i = 1:10
    yhat(:,i) = sum(yvals==i, 2) / K;
end
end