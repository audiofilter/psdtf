% define the number of samples 
M = 512;

% define the number of frames
N = 840;

% define the number of bases
K = 3;

% define the number of iterations
iterations = 200;

% execute source separation
[x, s, fs, bits] = kl_nmf_bss('../audio/mixture.wav', M, N, K, iterations);

% write audio files
for k = 1 : K
  wavwrite(s(:, k), fs, bits, ['source' num2str(k) '.wav']);
end

%% read ground truth
% g = zeros(length(x), K);
% g(:, 1) = wavread('../audio/C_part.wav');
% g(:, 2) = wavread('../audio/E_part.wav');
% g(:, 3) = wavread('../audio/G_part.wav');

%% evaluate performance
% [SDR, SIR, SAR, perm] = bss_eval_sources(s', g');
% fprintf(1, 'AverageSDR = %f [dB]\n', mean(SDR));
% fprintf(1, 'AverageSIR = %f [dB]\n', mean(SIR));
% fprintf(1, 'AverageSAR = %f [dB]\n', mean(SAR));
