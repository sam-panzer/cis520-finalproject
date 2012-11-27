function ranks = make_final_prediction(model, example)
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

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE

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
