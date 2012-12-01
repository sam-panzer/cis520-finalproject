% Train SVMs for Adaboost
N_svms = 4;
F = 3;

clear svms;
X_lyrics = make_lyrics_sparse(train, vocab);
X_lyrics = bsxfun(@rdivide, X_lyrics, sum(X_lyrics, 2));
Y = zeros(numel(train), 1);
for i=1:numel(train)
    Y(i) = genre_class(train(i).genre);
end

for i = 1:N_svms
    data_split = mod(randperm(n), F) == 0;
    curr = make_svm(X_lyrics(data_split,:), Y(data_split), @kernel_intersection);
    svms(i) = curr;
end

save('svms.mat', 'svms');