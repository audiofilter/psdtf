function Y = ensure_psd(X)

Y = 0.5 * (X + X');
Y = Y - eye(size(Y)) * min(min(eig(Y)), 0) + trace(Y) * eye(size(Y)) * 1.0e-6;
