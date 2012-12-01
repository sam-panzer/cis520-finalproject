function K = kernel_tanh(X, X2, k, c)
% Evaluates the Hyperbolic Tangent Kernel with specified k and c
%
% Usage:
%
%    K = KERNEL_TANH(X, X2, K, C)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the polynomial kernel
% with degree P.

% HINT: This should be a very straightforward one liner!

K = tanh(k*X*X2' + c);

% After you've computed K, make sure not a sparse matrix anymore
K = full(K)';