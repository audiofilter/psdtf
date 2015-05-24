function [cost, Y, H, W, index] = psdtf_mu_init(X, K, index)

M = size(X, 1);
N = size(X, 3);

H = zeros(N, K);
W = zeros(M, M, K);

for k = 1 : K
  for n = 1 : N
    H(n, k) = gamrnd(0.1, 1.0 / 0.1);
  end  
end

for k = 1 : K
  W(:, :, k) = wishrnd(eye(M) / M, M);
end

for k = 1 : K
  scale = trace(W(:, :, k));
  
  W(:, :, k) = W(:, :, k) / scale;
  H(:, k) = H(:, k) * scale;
end

[H, W, index] = permute_data(H, W, index);

Y = zeros(M, M, N);

for n = 1 : N
  for k = 1 : K
    Y(:, :, n) = Y(:, :, n) + H(n, k) * W(:, :, k);
  end
end

c = 0.0;

for n = 1 : N
  c = c - log(det(X(:, :, n) / Y(:, :, n)));
  c = c + trace(X(:, :, n) / Y(:, :, n));
  c = c - M;
end

cost(1) = c; 
fprintf(1, 'cost[%d] = %f\n', length(cost), c);
