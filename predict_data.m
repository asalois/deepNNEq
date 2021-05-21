function [data,symbols] = predict_data(SNR,nb)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% Add AWGN to the signal

%% Proakis Synthetic Channel Equilization

% Montana State University
% Electrical & Computer Engineering Department
% Created by Alexander Salois

% prelim comands
% clc;
% clear;
% close all;

% Modulated signal parameters
M = 4; % order of modulation

% Specify a seed for the random number generators to ensure repeatability.
% rng(12345)

% Generate a PSK signal
msg = randi([0 M-1],nb,1);
symbols = qammod(msg,M);

% Channel parameters
chnl = [0.407 0.815 0.407];

% Pass the signal through the channel
filtSig = filter(chnl,1,symbols);
filtSig = filtSig(2:end);
niosySig = awgn(filtSig,SNR,'measured');
inputSig = niosySig;

numSamples = 4;
data = makeInputMat(inputSig,numSamples);
data = data(:,1:end-numSamples);
data = data';
%savename = sprintf('predictSNR%02d',SNR)
%save(savename,'data')

end

