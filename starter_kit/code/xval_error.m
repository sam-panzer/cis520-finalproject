function [error evec] = xval_error(examples, vocab, n_folds)
% xval_error - cross-validation error.
%
% Usage:
%
%   ERROR = xval_error(examples, vocab, n_folds)
%
% Returns the average N-fold cross validation error of predict_genre on the 
% given dataset when the dataset is partitioned with n_folds folds.
%

N = size(examples, 2);
parts = mod(randperm(N), n_folds) + 1;
s = zeros(n_folds, 1);
Y = zeros(N, 1);
for i=1:N
    Y(i) = genre_class(examples(i).genre);
end
for i = 1:n_folds
    fprintf('Beginning crossvalidation step %d of %d folds.\n', i, n_folds);
    train_examples = examples(parts ~= i);
    train_model = make_model(train_examples, vocab);
    test_examples = examples(parts == i);
    Yq = Y(parts == i, :);
    %Yt = Y(parts ~= i, :); % Yt is computed by train_model
    testRanking = make_final_prediction(train_model, test_examples);
    s(i) = rank_loss(testRanking, Yq);
    fprintf('<: xval-error at step %d was %g :> \n', i, s(i));
end
error = sum(s) / n_folds;
evec = s;
