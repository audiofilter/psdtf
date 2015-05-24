function [cost, Y, H, W, index] = psdtf_mu_train(cost, Y, H, W, X, index)

M = size(X, 1);
N = size(X, 3);
K = size(W, 3);

% regularizer
e = eye(M) * 1.0e-8;

%%% update H %%%
for k = 1 : K
  for n = 1 : N
    H(n, k) = H(n, k) * sqrt(trace(W(:, :, k) / Y(:, :, n) * X(:, :, n) / Y(:, :, n)) / trace(W(:, :, k) / Y(:, :, n)));
  end  
end

[H, W, index] = permute_data(H, W, index);

% calculate Y
Y = zeros(M, M, N);

for n = 1 : N
  for k = 1 : K
    Y(:, :, n) = Y(:, :, n) + H(n, k) * W(:, :, k);
  end
  
  % ensure positive definiteness
  Y(:, :, n) = 0.5 * (Y(:, :, n) + Y(:, :, n)');
end

%%% update W %%%
for k = 1 : K
  A = zeros(M, M);
  B = zeros(M, M);
  
  for n = 1 : N
    A = A + H(n, k) * eye(M) / Y(:, :, n);
    B = B + H(n, k) * eye(M) / Y(:, :, n) * X(:, :, n) / Y(:, :, n);
  end  
  
  % ensure positive definiteness
  A = 0.5 * (A + A') + e;
  B = 0.5 * (B + B') + e;
  
  L = chol(B, 'lower');
  
  W(:, :, k) = W(:, :, k) * L / sqrtm(L' * W(:, :, k) * A * W(:, :, k) * L) * L' * W(:, :, k);
  
  % ensure positive definiteness
  W(:, :, k) = 0.5 * (W(:, :, k) + W(:, :, k)') + e;
end

% normalize W
for k = 1 : K
  scale = trace(W(:, :, k));
    
  W(:, :, k) = W(:, :, k) / scale;
  H(:, k) = H(:, k) * scale;
end

[H, W, index] = permute_data(H, W, index);

% calculate Y
Y = zeros(M, M, N);

for n = 1 : N
  for k = 1 : K
    Y(:, :, n) = Y(:, :, n) + H(n, k) * W(:, :, k);
  end
  
  % ensure positive definiteness
  Y(:, :, n) = 0.5 * (Y(:, :, n) + Y(:, :, n)');
end

%%% calculate cost %%%
c = 0.0;

for n = 1 : N
  c = c - log(det(X(:, :, n) / Y(:, :, n)));
  c = c + trace(X(:, :, n) / Y(:, :, n));
  c = c - M;
end

cost(length(cost) + 1) = c;   
fprintf(1, 'cost[%d] = %f\n', length(cost), c);
