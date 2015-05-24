function [cost, W, H, Y, XiY] = ld_psdtf_train(cost, W, H, Y, XiY, X, rank)

M = size(X, 1);
N = size(X, 3);
K = size(W, 3);

if nargin == 2
  rank = M;
end

% update H
for k = 1 : K
  for n = 1 : N
    V = W(:, :, k) / Y(:, :, n);
    H(n, k) = H(n, k) * sqrt(trace(V * XiY(:, :, n)) / trace(V));
  end  
end

% calculate Y
Y = zeros(M, M, N, 'single');

for n = 1 : N
  for k = 1 : K
    Y(:, :, n) = Y(:, :, n) + H(n, k) * W(:, :, k);
  end
  
  Y(:, :, n) = ensure_psd(Y(:, :, n));
  XiY(:, :, n) = X(:, :, n) / Y(:, :, n);
end

% update W
for k = 1 : K
  A = zeros(M, M);
  B = zeros(M, M);
  
  for n = 1 : N
    C = H(n, k) * eye(M) / Y(:, :, n);
    A = A + C;
    B = B + C * XiY(:, :, n);
  end
  
  A = ensure_psd(A);
  B = ensure_psd(B);
  
  L = chol(B, 'lower');
  
  W(:, :, k) = W(:, :, k) * L / sqrtm(ensure_psd(L' * W(:, :, k) * A * W(:, :, k) * L)) * L' * W(:, :, k);
  W(:, :, k) = ensure_psd(real(W(:, :, k)));
end

% normalize W
for k = 1 : K
  scale = trace(W(:, :, k));
  
  W(:, :, k) = W(:, :, k) / scale;
  H(:, k) = H(:, k) * scale;
end

% calculate Y
Y = zeros(M, M, N, 'single');

for n = 1 : N
  for k = 1 : K
    Y(:, :, n) = Y(:, :, n) + H(n, k) * W(:, :, k);
  end
  
  Y(:, :, n) = ensure_psd(Y(:, :, n));
  XiY(:, :, n) = X(:, :, n) / Y(:, :, n);
end

% calculate cost
c = 0.0;

for n = 1 : N
  if rank ~= M 
    ev = sort(eig(X(:, :, n)), 'descend');
    c = c - sum(log(ev(1 : rank)));
  else
    c = c - 2.0 * sum(log(diag(chol(X(:, :, n), 'lower'))));
  end
  
  c = c + 2.0 * sum(log(diag(chol(Y(:, :, n), 'lower'))));
  c = c + trace(XiY(:, :, n));
  c = c - M;
end

cost(length(cost) + 1) = c;
