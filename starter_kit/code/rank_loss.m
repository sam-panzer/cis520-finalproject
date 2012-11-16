function loss = rank_loss(predicted_ranks, Y)
    N = size(Y, 1);
    r = zeros(N, 1);
    indices = predicted_ranks == (repmat(Y, 1, size(predicted_ranks, 2)));
    for i = 1:N
        r(i) = 1 - 1 / find(indices(i,:));
    end
    loss = sum(r)/ N;
end