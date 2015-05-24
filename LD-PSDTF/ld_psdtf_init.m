function [cost, W, H, Y, XiY] = ld_psdtf_init(X, K, rank)

M = size(X, 1);
N = size(X, 3);

W   = zeros(M, M, K);
H   = zeros(N, K);
XiY = zeros(M, M, N, 'single');

if nargin == 2
  rank = M;
end

average = 0;

for n = 1 : N
  average = average + trace(X(:, :, n));
end

average = average / (M * N);

% initialize H
for k = 1 : K
  for n = 1 : N
    H(n, k) = gamrnd(1.0, average * M / K);
  end  
end

% initialize W
for k = 1 : K
  W(:, :, k) = wishrnd(eye(M), M) / M^2;
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

cost = zeros(1, 1);
cost(1) = c; 
