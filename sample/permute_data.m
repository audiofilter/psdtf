function [H, W, index] = permute_data(H, W, index)

K = size(W, 3);

total = zeros(K, 1);

for k = 1 : K
  total(k) = sum(H(:,k));
end

[temp, index2] = sort(total, 'descend');

H = H(:, index2);
W = W(:, :, index2);
index = index(index2);
