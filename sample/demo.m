rand('twister', sum(100 * clock));
iters = 500;

M = 10;
N = 1000;
K = 5;

% synthesize
[X, H_truth, W_truth] = synthesize_data(M, N, K);

% factorize
index = [1 : K];
[cost, Y, H, W, index] = psdtf_mu_init(X, K, index);

Y0 = Y;
H0 = H;
W0 = W;

for i = 1 : iters
  [cost, Y, H, W, index] = psdtf_mu_train(cost, Y, H, W, X, index);
  
  figure(1);
  for k = 1 : K
    subplot(2, K, k);
    imagesc(W_truth(:, :, k));
    %colormap('bone');
    axis square;
    
    subplot(2, K, K + k);
    imagesc(W(:, :, index(k)));
    %colormap('bone');
    axis square;
  end
end

% visualize
figure(2);
visualize_data(H_truth, W_truth);
figure(3);
visualize_data(H, W);
