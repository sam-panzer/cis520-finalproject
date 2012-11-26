function [error] = xval_error(X_lyrics, X_audio, Y, n_folds)
% xval_error - cross-validation error.
%
% Usage:
%
%   ERROR = xval_error(X_lyrics, X_audio, Y, n_folds)
%
% Returns the average N-fold cross validation error of predict_genre on the 
% given dataset when the dataset is partitioned with n_folds folds.
%

N = size(X_lyrics, 1);
parts = mod(randperm(N), n_folds) + 1;
s = zeros(n_folds, 1);
for i = 1:n_folds
    Xq_lyrics = X_lyrics(parts == i, :);
    Xt_lyrics = X_lyrics(parts ~= i, :);
    Xq_audio = X_audio(parts == i, :);
    Xt_audio = X_audio(parts ~= i, :);
    Yq = Y(parts == i, :);
    Yt = Y(parts ~= i, :);
    testRanking = predict_genre(Xt_lyrics, Xq_lyrics, Xt_audio, Xq_audio, Yt);
    s(i) = rank_loss(testRanking, Yq);
end
error = sum(s) / n_folds;
