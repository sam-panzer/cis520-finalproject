function [ranks] = make_final_prediction(model, examples, p)
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
lyrics = make_lyrics_sparse(example, model.vocab);
audio = make_audio(example);

% Find nearest neighbor
%D = model.Xt*X';
%[~,nn] = max(D);
%yhat = model.Yt(nn);
%scores = sqrt((dt_value_ranked(model.lyrics_tree,lyrics))) + sqrt((dt_value_ranked(model.audio_tree,audio))); 
%scores = sqrt((dt_value_ranked(model.lyrics_tree,lyrics))) + sqrt((dt_value_ranked(model.audio_tree,audio))); 
%scores = (eval_Stochastic_Bosque(lyrics,model.lyrics_forest)) + (audio,eval_Stochastic_Bosque(model.audio_forest)).^2; 
scores = (eval_Stochastic_Bosque(audio,model.audio_forest)); 

% Convert into score vector
%scores = zeros(1,10);
%scores(yhat) = 1;

% Convert into ranks
ranks = get_ranks(scores);
=======
X_lyrics = make_lyrics_sparse(examples, model.vocab);
X_audio = make_audio(examples);
Yt = zeros(N, 1);
for i=1:N
    Yt(i) = genre_class(examples(i).genre);
end

%K_lyrics = model.svm_lyrics.kernel(model.Xt_lyrics, X_lyrics);
K_audio = model.svm_audio.kernel(model.Xt_audio, X_audio);

[~, ~, probs_reordered_a] = svmpredict(Yt, [(1:size(K_audio,1))' K_audio], ...
    model.svm_audio.svm, '-b 1');
%[~, ~, probs_reordered_l] = svmpredict(Yt, ...
%    [(1:size(K_lyrics,1))' K_lyrics], model.svm_lyrics.svm, '-b 1');

% We need to permute the labels according to model.svm_lyrics.svm.Label;
% Why, libsvm? Why?
%labels_l = model.svm_lyrics.svm.Label;
labels_a = model.svm_audio.svm.Label;
%probs_l = zeros(N, 10);
probs_a = zeros(N, 10);
for i = 1:size(probs_reordered_a, 2);
%    probs_l(:, labels_l(i)) = probs_reordered_l(:, i);
    probs_a(:, labels_a(i)) = probs_reordered_a(:, i);
end

% KNN
% yhat_a = knn(model.Xt_audio, X_audio, model.Yt, model.K);
% yhat = knn(model.Xt_lyrics, X_lyrics, model.Yt, model.K);

% Convert into score vector
% w = model.weights;

ranks = get_ranks(probs_a);
