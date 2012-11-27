function [model] = make_model(train, vocab)

% Used for all models
[Xt_lyrics] = make_lyrics_sparse(train, vocab);
Xt_audio = make_audio(train);

Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end
Xt_lyrics = bsxfun(@rdivide, Xt_lyrics, sum(Xt_lyrics, 2));
Nlabels = max(Yt);
priors = zeros(Nlabels, 1);
for i = 1:Nlabels
  priors(i) = sum(Yt == i);
end
model.priors = priors / sum(priors);
model.vocab = vocab;

model.Xt_lyrics = Xt_lyrics;
model.Xt_audio = Xt_audio;

% Used only for KNN
% model.Xt_lyrics = Xt_lyrics;
model.Xt_audio = Xt_audio;
model.Yt = Yt;

model.K = 153;
% model.weights = [0.65, 0.15, 0.2];

% Used only for SVM
model.weights = [0.25, 0.75, 0];
%'Making audio model...'
%model.svm_audio = make_svm(Xt_audio, Yt, @(x, x2) kernel_poly(x, x2, 1));
'Making lyrics model...'
model.svm_lyrics = make_svm(Xt_lyrics, Yt, @kernel_intersection);
end