% THIS IS NOT A FUNCTION.
%
% This is just a script that is called to train the model. It saves
% whatever we will need at test time.

load('../data/music_dataset.mat');

% Generate the sparse training set that we'll need for nearest neighbor
model = make_model(train, vocab);

save('my_model.mat', 'model');
