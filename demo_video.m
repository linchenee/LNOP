clc;
clear;
close all;
addpath(genpath('test_video'));
addpath(genpath('Utilities'));

load('hall_qcif.mat');  % groundtruth

%% Scenario generation
[n1,n2,n3] = size(hall_qcif);
SNR = 5;  % SNR=5dB
Rho = 1 / (10^(SNR / 10));
hall_noisy = zeros(n1,n2,n3);  % noisy video 
for i = 1 : n3
 hall_noisy(:,:,i) = double(imnoise(uint8(hall_qcif(:,:,i)),'salt & pepper', Rho));  
end
Sampling_rate = 0.25;
Omega = find(rand(n1*n2*n3,1) < Sampling_rate);  % locations of the known entries
Y = zeros(n1,n2,n3);  % noisy and incomplete video
Y(Omega) = hall_noisy(Omega);
Noises = hall_noisy(Omega) - hall_qcif(Omega);

%% LNOP (p=1) algorithm for video restoration
opts.p = 1; 
opts.epsilon = 6e2;
opts.lambda = 1e7;
opts.indicator = 1 * norm(Noises, opts.p);
opts.iter = 250; 
opts.mu = 1e-3;   
opts.mu_max = 1e5;   
opts.rho = 1.1;  
X1 = LNOP(Y, Omega, opts);

Mask = ones(n1,n2,n3);  % Calculate the PSNR on the locations of unknown entries.
Mask(Omega) = 0;
PSNR1 = PSNR(hall_qcif, X1, Mask);
fprintf('PSNR achieved by LNOP (p=1) is %d dB\n', PSNR1);
figure(1);
imshow(uint8(X1(:,:,1)), []);

%% LNOP (p=0.7) algorithm for video restoration
opts.p = 0.7; 
opts.epsilon = 5e2;
opts.lambda = 1e7;
opts.indicator = 1 * norm(Noises, opts.p);
opts.iter = 250; 
opts.mu = 1e-3;   
opts.mu_max = 1e5;   
opts.rho = 1.1;  
X2 = LNOP(Y, Omega, opts);

PSNR2 = PSNR(hall_qcif, X2, Mask);
fprintf('PSNR achieved by LNOP (p=0.7) is %d dB\n', PSNR2);
figure(2);
imshow(uint8(X2(:,:,1)), []);
