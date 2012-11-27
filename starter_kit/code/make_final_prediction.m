function [ranks] = make_final_prediction(model, examples)
% Uses your trained model to make a final prediction for a SINGLE example.
%
% Usage:
%
%   RANKS = MAKE_FINAL_PREDICTION(MODEL, EXAMPLE);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes your trained model (output from INIT_MODEL) and a SINGLE 
% example, and returns a ranking ROW VECTOR as explained in the project
% overview.
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Your model should be loaded from disk in INIT_MODEL. DO NOT DO
% ANY TRAINING HERE.

N = size(examples, 2);
% We only take in one example at a time.
X_lyrics = make_lyrics_sparse(examples, model.vocab);
X_audio = make_audio(examples);


K_lyrics = model.svm_lyrics.kernel(model.Xt_lyrics, X_lyrics);
%K_audio = kernel(X_audio, X_audio);

%[yhat_a , ~, ~] = svmpredict((1:size(K_audio, 1))', [(1:size(K_audio,1))' K_audio], model.svm_audio.svm);
[yhat_l , ~, ~] = svmpredict((1:size(K_lyrics, 1))', ...
    [(1:size(K_lyrics,1))' K_lyrics], model.svm_lyrics.svm);
% KNN
 yhat_a = knn(model.Xt_audio, X_audio, model.Yt, model.K);
% yhat = knn(model.Xt_lyrics, X_lyrics, model.Yt, model.K);

% Convert into score vector
% w = model.weights;
ya = get_ranks(yhat_a);
rest = reshape(ya(ya ~= repmat(yhat_l, 1, 10)), N, 9);
ranks = [yhat_l rest];

