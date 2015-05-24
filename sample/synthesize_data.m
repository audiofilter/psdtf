function [X, H, W] = synthesize_data(M, N, K)

X = zeros(M, M, N);
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

index = [1 : K];
[H, W, index] = permute_data(H, W, index);

for n = 1 : N
  for k = 1 : K
    X(:, :, n) = X(:, :, n) + H(n, k) * W(:, :, k);
  end
  
  X(:, :, n) = wishrnd(X(:, :, n), M);
end
