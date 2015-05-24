function visualize_data(H, W)

K = size(W, 3);

for k = 1 : K
  subplot(K, 2, 2 * k - 1);
  imagesc(W(:,:,k));  
  axis square  
  %colormap('bone');
  
  subplot(K, 2, 2 * k);
  plot(H(:,k));
end
