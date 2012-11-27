% THIS IS NOT A FUNCTION.
%
% This is just a script that is called to train the model. It saves
% whatever we will need at test time.

load('../data/music_dataset.mat');

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE

% Generate the sparse training set that we'll need for nearest neighbor
[Xt_lyrics] = make_lyrics_sparse(train, vocab);
Xt_audio = make_audio(train);

Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end
Xt = bsxfun(@rdivide, Xt_lyrics, sum(Xt_lyrics, 2));

model.vocab = vocab;
model.Xt = Xt;
model.Yt = Yt;

lyrics
model.random_forest = Stochastic_Bosque(Xt_lyrics(1:8000,:),Yt(1:8000,:),'ntrees',20,'depth',6);

save('my_model.mat', 'model');