function nb = nb_train(X, Y)
% Straight out of homework 4.
% Train a Gaussian Naive Bayes model with shared variances.
%
% Usage:
%
%   [NB] = NB_TRAIN(X, Y)
%
% X is a N x P matrix of N examples with P features each. Y is a N x 1 vector
% of 0-1 class labels. Returns a struct NB with fields:
%    nb.p_y          -- vector, P(Y=k) for each 1 <= k <= K
%    nb.mu_x_given_y -- P x K matrix of class means for each feature
%    nb.sigma_x      -- P x 1 matrix of standard deviations for each feature
% 
% SEE ALSO
%   NB_TEST

% **** NOTE: Variances should never be zero, even if the variance of the
% data is zero. Therefore you should always add a small positive constant
% to estimates of variance to prevent your prediction code from crashing.
% Use the matlab constant 'eps' for this.

K = max(Y);
[N P] = size(X);
counts = zeros(K, 1);
sigmas = zeros(size(X,2), K);
for i = 1:K
  counts(i) = sum(Y == i);
end
nb.mu_x_given_y = zeros(P, K);
nb.p_y = zeros(1,K);
for i = 1:K
    nb.p_y(i) = sum(Y == i) / N;
    nb.mu_x_given_y(:,i) = mean(X(Y == i, :));
    sigmas(:, i) = var(X(Y == i, :), 1) * counts(i);
end
% It took me *way* too long to remember the sqrt here...
nb.sigma_x = sqrt(eps + sum(sigmas, 2) / N);
end