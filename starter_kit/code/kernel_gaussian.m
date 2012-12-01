function K = kernel_gaussian(X, X2, sigma)
% Evaluates the Gaussian Kernel with specified sigma
%
% Usage:
%
%    K = KERNEL_GAUSSIAN(X, X2, SIGMA)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the Guassian kernel
% with parameter sigma=20.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.
X = X';
X2 = X2';

% YOUR CODE GOES HERE.

t = CTimeleft(m);
for i = 1:m
    t.timeleft();
    for j = 1:n
        K(i,j) = (norm(X2(:, i) - X(:, j)))^2;
    end
end
K = full(exp(-K / (2*sigma^2)));