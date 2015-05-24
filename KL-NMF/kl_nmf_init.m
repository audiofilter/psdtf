function [cost, W, H, Y, XiY] = kl_nmf_init(X, K)

M = size(X, 1);
N = size(X, 2);

W   = zeros(M, K);
H   = zeros(N, K);
XiY = zeros(M, N);

average = mean(mean(X));

% initialize H
for k = 1 : K
  for n = 1 : N
    H(n, k) = gamrnd(1.0, average * M / K);
  end  
end

% initialize W
for k = 1 : K
  for m = 1 : M
    W(m, k) = gamrnd(1.0, 1.0 / M);
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

cost = zeros(1, 1);
cost(1) = c; 
