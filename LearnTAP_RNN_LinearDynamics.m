% Script to embed TAP dynamics into neural response space as an
% RNN. After emedding the dynamics, the next step is to learn the embedding
% from data and subsequently learn the parameters of the inference
% algorithm.

clear;
load Data/KTrue;

NVars = 2; % No. of variables of x
% JMat  = GenJMat(NVars)/3; % Generate the coupling matrix J
c = -0.3;
JMat = (1-c)*eye(NVars) + c;

JTrue       = JMat;
JTrueVec    = JMatToVec(JTrue);

% Parameters for TAP Dynamics
N_H = 500; % No. of sessions. Each session has a different input
N_T = 30;  % No. of time steps per session
lam = 0.1; % Relaxation term

hMat = 1*randn(NVars, N_H); % Generate the inputs for each session

xMat  = zeros(NVars,N_T,N_H);
A = randn(NVars,NVars);
[A,~,~] = svd(A);
A = A*0;

for hh = 1:N_H
    
    hVec    = hMat(:,hh);
    rOld    = 0*randn(NVars,1); % Initialize 
    r       = zeros(NVars,1);

    for nIter = 1:N_T
        r = (1-lam)*rOld + lam*(A*rOld - 0.1*hVec);
        xMat(:,nIter,hh) = r;
        rOld = r;
    end

end




% Now construct the feedforward neural network

% gain constants for the hidden layer weights and bias terms
% NNeu = 100*NVars; % No. of neurons in the hidden layer. We need a state expansion
NNeu = 3; % No. of neurons in the hidden layer. We need a state expansion
g1 = 1/sqrt(NNeu);
g2 = 1/sqrt(NNeu);
g3 = 1/sqrt(NNeu);


% Weight and biases
W1 = g1*randn(NNeu,NVars);
W2 = g2*randn(NNeu,NVars);
B  = g3*randn(NNeu,1);


% Concatenate the outputs and hidden layer activations for each session
% into one big matrix
OutFull = [];
HidFull = [];

for hh = 1:N_H
    InputMat    = repmat(hMat(:,hh), 1, N_T);
    xCurrMat    = [zeros(NVars,1), xMat(:,1:N_T-1,hh)]; % adding zeroinput also
    xNextMat    = xMat(:,:,hh);
    BiasMat     = repmat(B,1,N_T);
    SigInp      = W1*xCurrMat + W2*InputMat + BiasMat;
    % HiddenMat   = 1./(1+exp(-SigInp));   
    HiddenMat   = SigInp;   
    
    HidFull     = [HidFull, HiddenMat];
    OutFull     = [OutFull, xNextMat];

end

% V = OutFull*pinv(HidFull);
% V = OutFull'\HidFull';
V = transpose(HidFull'\OutFull');

% Now generate the neural dynamics

rNeuMat     = zeros(NNeu,N_T,N_H);
rNeuMat_d   = zeros(NNeu,N_T,N_H); % direct embedding using V
for hh = 1:N_H
    rOld    = zeros(NNeu,1);
    for tt = 1:N_T
        SigInp  = W1*V*rOld + W2*hMat(:,hh) + B;
        rNew    = 1./(1+exp(-SigInp));
        
        
        rNeuMat(:,tt,hh) = rNew;
        rOld = rNew;
    end
    rNeuMat_d(:,:,hh) = pinv(V)*xMat(:,:,hh);
end


% Now to learn the projections from data
Rinit           = zeros(NNeu,N_H);
Rinit_d         = zeros(NNeu,N_H);
xinit           = zeros(NVars,N_H);
xhat_init_true  = zeros(NVars,N_H); % initial x obtained using true decoder

for hh = 1:N_H
    Rinit(:,hh)             = rNeuMat(:,1,hh); % Just time-step 1 is recorded
    Rinit_d(:,hh)           = rNeuMat_d(:,1,hh);
    xinit(:,hh)             = xMat(:,1,hh);
    xhat_init_true(:,hh)    = V*rNeuMat(:,1,hh);
end

Vhat = xinit'\Rinit';

