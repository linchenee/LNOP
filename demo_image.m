clc;
clear;
close all;
addpath(genpath('test_image'));
addpath(genpath('Utilities'));

I1 = double(imread('Image.jpg'));  % groundtruth

%% Scenario generation
[n1,n2,n3] = size(I1);
SNR = 3;  % SNR=3dB
Rho = 1 / (10^(SNR / 10));
I2 = double(imnoise(uint8(I1),'salt & pepper', Rho));  % noisy image  
Sampling_rate = 0.5;
Omega = find(rand(n1*n2*n3,1) < Sampling_rate);  % locations of the known entries
Y = zeros(n1,n2,n3);  % noisy and incomplete image
Y(Omega) = I2(Omega);
Noises = I2(Omega) - I1(Omega);

%% LNOP (p=1) algorithm for image restoration
opts.p = 1; 
opts.epsilon = 9e3;
opts.lambda = 1e5;
opts.indicator = 1 * norm(Noises, opts.p);
opts.iter = 100; 
opts.mu = 1e-3;   
opts.mu_max = 1e5;   
opts.rho = 1.1; 
X1 = LNOP(Y, Omega, opts);

Mask = ones(n1,n2,n3);  % Calculate the PSNR on the locations of unknown entries.
Mask(Omega) = 0;
PSNR1 = PSNR(I1, X1, Mask);
fprintf('PSNR achieved by LNOP (p=1) is %d dB\n', PSNR1);
figure(1);
imshow(uint8(X1), []);

%% LNOP (p=0.7) algorithm for image restoration
opts.p = 0.7; 
opts.epsilon = 2e3;
opts.lambda = 1e7;
opts.indicator = 1 * norm(Noises, opts.p);
opts.iter = 100; 
opts.mu = 1e-3;   
opts.mu_max = 1e5;   
opts.rho = 1.1; 
X2 = LNOP(Y, Omega, opts);

PSNR2 = PSNR(I1, X2, Mask);
fprintf('PSNR achieved by LNOP (p=0.7) is %d dB\n', PSNR2);
figure(2);
imshow(uint8(X2), []);
