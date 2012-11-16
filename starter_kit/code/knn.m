function [ yhat ] = knn(Xt_raw, Xq_raw, Yt, K)
% knn Summary of this function goes here
%   Detailed explanation goes here
Xt = bsxfun(@rdivide, Xt_raw, sum(Xt_raw, 2));
Xq = bsxfun(@rdivide, Xq_raw, sum(Xq_raw, 2));

D = Xq*Xt';
[~, idx] = sort(D, 2, 'descend');
ynn = idx(:, 1:K);
yhat = mode(Yt(ynn), 2);
end