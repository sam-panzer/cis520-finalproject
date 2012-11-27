function ranks = predict_genre(Xt_lyrics, Xq_lyrics, ...
                               Xt_audio, Xq_audio, ...
                               Yt)
% Returns the predicted rankings, given lyric and audio features.
%
% Usage:
%
%   RANKS = PREDICT_GENRE(XT_LYRICS, YT_LYRICS, XQ_LYRICS, ...
%                         XT_AUDIO, YT_AUDIO, XQ_AUDIO);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes a set of lyric and audio features and produces a
% ranking matrix as explained in the project overview. 
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Therefore, you should train your model BEFORE submission, save
% it in a .mat file, and load it here.

load forests_model_d10.mat;

ranks = zeros(size(Xq_lyrics,1),10);
for example=1:size(Xq_lyrics,1)
    
    %scores = sqrt((dt_value_ranked(model.lyrics_tree,Xq_lyrics(example,:)))) + sqrt((dt_value_ranked(model.audio_tree,Xq_audio(example,:)))); 
    scores = eval_Stochastic_Bosque(Xq_audio(example,:),model.audio_forest);
    ranks(example,:) = get_ranks(scores);
    
end

% K = 153; % Selected by cross-validation
% w = [0.65, 0.15, 0.2];
% Nlabels = max(Yt);
% priors = zeros(Nlabels, 1);
% for i = 1:Nlabels
%   priors(i) = sum(Yt == i);
% end
% 
% priors = w(3) * priors / sum(priors);
% 
% N = size(Xq_lyrics, 1);
% %scores = repmat(priors', N, 1);
% %scores_a = repmat(priors', N, 1);
% 
% % nbta = nb_train(Xt_audio, Yt);
% % yhat_a = nb_test(nbta, Xq_audio, 10);
% yhat_a = knn(Xt_audio, Xq_audio, Yt, K);
% 
% %nbt = nb_train(Xt_lyrics, Yt);
% %yhat = nb_test(nbt, Xq_lyrics, 10);
% yhat = knn(Xt_lyrics, Xq_lyrics, Yt, K);
% 
% %for i=1:N
% %    scores(i, yhat(i)) = 1 + scores(yhat(i));
% %    scores_a(i, yhat_a(i)) = 1 + scores(yhat_a(i));
% %end
% 
% ranks = get_ranks(w(1) * yhat + w(2) * yhat_a + w(3) * repmat(priors', N, 1));

end
