function [cost, W, H, Y, XiY] = kl_nmf_train(cost, W, H, Y, XiY, X)

M = size(X, 1);
N = size(X, 2);
K = size(W, 2);

% update H
for k = 1 : K
  for n = 1 : N
    H(n, k) = H(n, k) * (sum(W(:, k) .* XiY(:, n)) / sum(W(:, k)));
  end
end

% calculate Y
Y = zeros(M, N);

for n = 1 : N
  for k = 1 : K
    Y(:, n) = Y(:, n) + H(n, k) * W(:, k);
  end
  
  XiY(:, n) = X(:, n) ./ Y(:, n);
end

% update W
for k = 1 : K
  for m = 1 : M
    W(m, k) = W(m, k) * (sum(H(:, k) .* XiY(m, :)') / sum(H(:, k)));
  end
end

% normalize W
for k = 1 : K
  scale = sum(W(:, k));
  
  W(:, k) = W(:, k) / scale;
  H(:, k) = H(:, k) * scale;
end

% calculate Y
Y = zeros(M, N);

for n = 1 : N
  for k = 1 : K
    Y(:, n) = Y(:, n) + H(n, k) * W(:, k);
  end
  
  XiY(:, n) = X(:, n) ./ Y(:, n);
end

% calculate cost
c = 0.0;

for n = 1 : N
  c = sum(X(:, n) .* log(XiY(:, n)));
  c = c - sum(X(:, n));
  c = c + sum(Y(:, n));
end

cost(length(cost) + 1) = c;   
