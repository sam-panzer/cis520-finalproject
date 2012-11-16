function ranks = predict_genre(Xt_lyrics, Xq_lyrics, ...
                               Xt_audio, Xq_audio, ...
                               Yt, K, w)
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

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE

N = size(Xq_lyrics, 1);
scores_a = zeros(N, 10);

yhat_a = knn(Xt_audio, Xq_audio, Yt, K(2));
% K for lyrics: 83
for i=1:N
    scores_a(i, yhat_a(i)) = 1;
end

ranks = get_ranks( scores_a);

end
