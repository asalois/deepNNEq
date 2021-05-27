%% Proakis Synthetic Channel Equilization with Deep NNs

% Montana State University
% Electrical & Computer Engineering Department
% Created by Alexander Salois

% prelim comands
clc; clear; close all
tic
ber = zeros(1,39);
%%
for i = 1:39
    lname = sprintf('deepSNR%02d.mat',i);
    load(lname)
    rname = sprintf('predictionsSNR%02d.csv',i);
    pred = readmatrix(rname);
    eq = pred(:,2)+pred(:,3)*1i;
    x_b = qamdemod(seq,4);
    x_b = x_b(5:end-5);
    y_b = qamdemod(eq,4);
    y_b = y_b(1:end-4);
    [~, ber_dnn] = biterr(x_b,y_b)
    ber(i) = ber_dnn;
end
toc
%%
snr = 1:39;
figure()
semilogy(snr,ber,'-*')
save('berDNNTF','ber')