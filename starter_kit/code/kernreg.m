function [ yhat ] = kernreg(Xt_raw, Xq_raw, Yt, sigma)
% kernreg Summary of this function goes here
%   Detailed explanation goes here
Xt = bsxfun(@rdivide, Xt_raw, sum(Xt_raw, 2));
Xq = bsxfun(@rdivide, Xq_raw, sum(Xq_raw, 2));

D = Xq*Xt'; %Note I think D is actually D^2?

kernMat = exp(-D/(sigma.^2));

votes = zeros(size(Xq_raw,1),10);

for i = 1:10
    votes(i) = sum(kernMat(Yt==i, :));
end

% By this point votes is an M x 10 matrix such that votes(i,j) should be
% the weighted vote that example i is labeled j

[~, yhat] = sort(votes, 2, 'descend');

end