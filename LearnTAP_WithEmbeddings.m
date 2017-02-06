% Script to learn the parameters of the TAP Inference

clear;
load Data/KTrue;

NVars = 5;
JMat  = GenJMat(NVars);

JTrue = JMat;

JTrueVec = JMatToVec(JTrue);

% Run the TAP Inference
N_T = 30;
N_H = 500;
lam = 0.1;

hMat = 1*randn(NVars, N_H);

rMat = RunTAP(JMat, N_T, N_H, hMat, lam);


% Embed this into neural activity

% First create the orthogonal embedding matrix U
NNeu = NVars*1000; % No. of neurons
% TempMat = randn(NNeu,NNeu);
% TempMat = TempMat + TempMat';
% [EigVecs,~] = eig(TempMat);
% U = EigVecs(:,1:NVars);
% clear TempMat EigVecs
U = randn(NNeu,NVars);


rNeuMat2 = RunNeuTAP(JMat, N_T, N_H, hMat, U, lam, 0.1);

% % Simple embedding! 
% 
% % Now, embed the dynamics into neural response space
% rNeuMat = zeros(NNeu,N_T,N_H);
% for hh = 1:N_H
%     rNeuMat(:,:,hh) = U*rMat(:,:,hh);
% end
% 
% % Now add noise 
% % rNeuMat = rNeuMat + 0.025*randn(NNeu,N_T,N_H); %the standard deviation is of the neural response
% % so SNR must be 1
% rNeuMat_Noisy = rNeuMat + 0.0*randn(NNeu,N_T,N_H);

% Learn the embedding matrix
% First construct all the required matrices


% Simple embedding case
rNeuMat_Noisy = rNeuMat2;

R = zeros(NNeu,N_H);
for hh = 1:N_H
    R(:,hh) = rNeuMat_Noisy(:,1,hh); % Just time 1 is being recorded
end

H  = 1./(1 + exp(-hMat)); % sigmoid(hMat)

Uhat = R*H'*inv(H*H')/lam;

% Now that we have learnt Uhat, let us learn rhat
rMat_Noisy = zeros(NVars,N_T,N_H);
for hh = 1:N_H
    rMat_Noisy(:,:,hh) = inv(Uhat'*Uhat)*Uhat'*rNeuMat_Noisy(:,:,hh);
end

% Replace the rMat with the noisy one
figure; subplot(2,1,1); plot(rMat(:,:,1)','bx-')
hold on
plot(rMat_Noisy(:,:,1)','ro-')
subplot(2,1,2); plot(rNeuMat2(1:20,:,1)','bx-')

rMat = rMat_Noisy;

% Generate the fMat
fMat = GenfMat(rMat,hMat,lam);
if any(isinf(fMat(:)))
    disp('fMat has inf');
    keyboard;
end

% Save data only for a few sessions, so that you can run Gradient Descent
% faster

rMat = rMat(:,:,1:5);
fMat = fMat(:,:,1:5);

save Data/rMat rMat; 
save Data/JMat JMat;
save Data/fMat fMat;