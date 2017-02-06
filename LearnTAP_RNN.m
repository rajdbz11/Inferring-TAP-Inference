% Script to first embed TAP dynamics into neural response space using an
% RNN. After emedding the dynamics, the next step is to learn the embedding
% from data and subsequently learn the parameters of the inference
% algorithm.

clear;
load Data/KTrue;

NVars = 5; % No. of variables of x
JMat  = GenJMat(NVars); % Generate the coupling matrix J

JTrue       = JMat;
JTrueVec    = JMatToVec(JTrue);

% Parameters for TAP Dynamics
N_H = 1000; % No. of sessions. Each session has a different input
N_T = 30;  % No. of time steps per session
lam = 0.1; % Relaxation term

hMat = 1*randn(NVars, N_H); % Generate the inputs for each session

% Run the TAP dynamics
xMat = RunTAP(JMat, N_T, N_H, hMat, lam);

% Embed this into neural response space
N_E = NVars*5;      % No. of neurons
U = randn(N_E,NVars);  % Use a random embedding matrix for now
rNeuMat = zeros(N_E,N_T,N_H);
pMat    = zeros(N_E,N_T,N_H); %inputs also embedded and specified explicitly for all time steps
for hh = 1:N_H
    rNeuMat(:,:,hh) = U*xMat(:,:,hh);
    pMat(:,:,hh) = U*repmat(hMat(:,hh),1,N_T);
end


% Now construct the feedforward neural network

% gain constants for the hidden layer weights and bias terms
NNeu = 10*N_E; % No. of neurons in the hidden layer. We need a state expansion
g1 = 1/sqrt(NNeu);
g2 = 1/sqrt(NNeu);
g3 = 1/sqrt(NNeu);


% Weight and biases
Wr = g1*randn(NNeu,N_E);
Wp = g2*randn(NNeu,N_E);
Wh = g2*randn(NNeu,NVars);
B  = g3*randn(NNeu,1);

% Concatenate the outputs and hidden layer activations for each sessions
% into one big matrix
YFull = [];
FFull = [];

for hh = 1:N_H
    % rt = rNeuMat(:,1:N_T-1,hh);
    % pt = pMat(:,1:N_T-1,hh);    
    % ft = 1./( 1 + exp(-( Wr*rt + Wp*pt + repmat(B,1,N_T-1) )) );
    % yt = rNeuMat(:,2:N_T,hh);
    rt = [zeros(N_E,1),rNeuMat(:,1:N_T-1,hh)];
    pt = pMat(:,1:N_T,hh);
    ft = 1./( 1 + exp(-( Wr*rt + Wp*pt + repmat(B,1,N_T) )) );
    % ft = tanh( Wr*rt + Wp*pt + repmat(B,1,N_T) );
    % ht = repmat(hMat(:,hh),1,N_T);
    % ft = 1./( 1 + exp(-( Wr*rt + Wh*ht + repmat(B,1,N_T) )) );
    yt = rNeuMat(:,1:N_T,hh);
    
    FFull = [FFull, ft];
    YFull = [YFull, yt];
end

% Least squares solution to find embedding matrix V for the RNN
% V = YFull*FFull'*inv(FFull*FFull');
V = YFull*pinv(FFull);

% YHatMat = zeros(N_E,N_T,N_H);
% RHatMat = zeros(NNeu,N_T,N_H);
% % Generating the predictions for each session
% for hh = 1:N_H
%     rt = [zeros(N_E,1),rNeuMat(:,1:N_T-1,hh)];
%     pt = pMat(:,1:N_T,hh);
%     ft = 1./( 1 + exp(-( Wr*rt + Wp*pt + repmat(B,1,N_T) )) );
%     % ht = repmat(hMat(:,hh),1,N_T);
%     % ft = 1./( 1 + exp(-( Wr*rt + Wh*ht + repmat(B,1,N_T) )) );
%     RHatMat(:,:,hh) = ft;
%     YHatMat(:,:,hh) = V*ft;
% end
    


% Now generate the neural dynamics in terms of the RNN

% U2 = inv(V'*V)*V'; % essentially a second embedding 
U2 = pinv(V);

% Simple way is just by using the embedding matrix
rNeu_RNN_Mat = zeros(NNeu,N_T,N_H);
for hh = 1:N_H
    rNeu_RNN_Mat(:,:,hh) = U2*rNeuMat(:,:,hh);
end

% The right way to do it is to run the RNN dynamics
rNeu_RNN_Mat2 = zeros(NNeu,N_T,N_H);
for hh = 1:N_H
    pVec = U*hMat(:,hh);
    hVec = hMat(:,hh);
    rOld = zeros(NNeu,1);
    for tt = 1:N_T
        argm = Wr*V*rOld + Wp*pVec + B;
        % argm = Wr*V*rOld + Wh*hVec + B;
        rNew = 1./(1 + exp(-argm));
        % rNew = tanh(argm);
        rNeu_RNN_Mat2(:,tt,hh) = rNew;
        rOld = rNew;
    end
end

DecoderMat = pinv(U)*V;
xMat1 = xMat(:,:,1);
xhat1 = DecoderMat*rNeu_RNN_Mat2(:,:,1);
figure; plot(xMat1','bx-'); hold on
plot(xhat1','ro-')

% % Now learn the combined embedding matrix: U2xU
% 
% R = zeros(NNeu,N_H);
% for hh = 1:N_H
%     % R(:,hh) = rNeu_RNN_Mat(:,1,hh); % Just time-step 1 is recorded
%     R(:,hh) = rNeu_RNN_Mat2(:,1,hh); % Just time-step 1 is recorded
% end
%  
% H  = 1./(1 + exp(-hMat)); % sigmoid(hMat)
%  
% % Uhat = R*H'*inv(H*H')/lam;
% Uhat = R*(pinv(H))/lam;
% 
% UCombined = U2*U; % This is the correct embedding matrix
% 
% % Now recover the dynamics in the variables
% rMat = xMat*0;
% for hh = 1:N_H
%     % rMat(:,:,hh) = inv(Uhat'*Uhat)*Uhat'*rNeu_RNN_Mat(:,:,hh);
%     rMat(:,:,hh) = pinv(Uhat)*rNeu_RNN_Mat2(:,:,hh);
% end
% 
% 
% DataSet = [];
% 
% for kk = 1:500
%     DataSet = [DataSet, rNeu_RNN_Mat(:,:,kk)];
% end
% DataSet = DataSet';
% [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(DataSet);
% 
