% THIS IS NOT A FUNCTION.
%
% This is just a script that is called to train the model. It saves
% whatever we will need at test time.

load('../data/music_dataset.mat');

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

%model.random_forest = Stochastic_Bosque(Xt_lyrics(1:5000,:),Yt(1:5000,:),'ntrees',17,'depth',5);
%model.audio_tree = dt_train_multi(Xt_audio,Yt,15);

%save('audio_tree_d20.mat', 'model');

%model.audio_forest = Stochastic_Bosque(Xt_audio,Yt,10);
model.audio_forest = Stochastic_Bosque(Xt_audio,Yt,'ntrees',50,'depth',12);
%model.lyrics_forest_test = Stochastic_Bosque(Xt_lyrics(1:5000,:),Yt(1:5000),'ntrees',2,'depth',2);

save('forests_model.mat', 'model');

%P = [];
%for i = 8001 : size(Yt,1)
%    P(i - 8000) = dt_value(model.audio_tree,Xt_audio(i,:));
%end
%sum(P' == Yt(8001:size(Yt,1),:)) / (size(Yt,1) - 8001)
