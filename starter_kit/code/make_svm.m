function [ model ] = make_svm( X, Y, kernel)
%make_svm Train an SVM for Xt, Yt
% Compute kernel matrices for training and testing.
K = kernel(X, X);

% Use built-in libsvm cross validation to choose the C regularization
% parameter.
crange = 10.^(0:1:4);
acc = zeros(numel(crange), 1);
for i = 1:numel(crange)
    acc(i) = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

% Train and evaluate SVM classifier using libsvm
model.svm = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange(bestc)));
model.K = K;
model.kernel = kernel;
end