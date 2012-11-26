function [y] = nb_test(nb, X, K)
% Generate predictions for a Gaussian Naive Bayes model.
%
% Usage:
%
%   [Y] = NB_TEST(NB, X)
%
% X is a N x P matrix of N examples with P features each, and NB is a struct
% from the training routine NB_TRAIN. Generates predictions for each of the
% N examples and returns a 0-1 N x 1 vector Y.
% 
% SEE ALSO
%   NB_TRAIN

[N P] = size(X);
log_p_x_and_y = zeros(N, K);
for i = 1:K
    log_p_x_and_y(:,i) = log(nb.p_y(i)) + sum(log( ...
    normpdf(X, repmat(nb.mu_x_given_y(:,i)', N, 1), ...
            repmat(nb.sigma_x', N, 1))), 2);
end

% Take the maximum of the log generative probability 
[~, y] = max(log_p_x_and_y, [], 2);
