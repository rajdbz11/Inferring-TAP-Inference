% Script to embed TAP dynamics into neural response space as an
% RNN. After emedding the dynamics, the next step is to learn the embedding
% from data and subsequently learn the parameters of the inference
% algorithm.

clear;
load Data/KTrue;

NVars = 3; % No. of variables of x
% JMat  = GenJMat(NVars)/3; % Generate the coupling matrix J
c = -0.3;
JMat = (1-c)*eye(3) + c;

JTrue       = JMat;
JTrueVec    = JMatToVec(JTrue);

% Parameters for TAP Dynamics
N_H = 1000; % No. of sessions. Each session has a different input
N_T = 30;  % No. of time steps per session
lam = 0.03; % Relaxation term

hMat = 1*randn(NVars, N_H); % Generate the inputs for each session

tic;
% Run the TAP dynamics
xMat = RunTAP(JMat, N_T, N_H, hMat, lam);
toc;

% Now construct the feedforward neural network

% gain constants for the hidden layer weights and bias terms
NNeu = 100*NVars; % No. of neurons in the hidden layer. We need a state expansion
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
    HiddenMat   = 1./(1+exp(-SigInp));
    HiddenMat   = HiddenMat + rand(size(HiddenMat));
    
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
        
        % add noise
        % rNew = rNew + 0.0005*randn(NNeu,1);
        
        rNeuMat(:,tt,hh) = rNew;
        rOld = rNew;
    end
    rNeuMat_d(:,:,hh) = pinv(V)*xMat(:,:,hh);
end

% rNeuMat = rNeuMat + 0.001*randn(size(rNeuMat));

% Now to learn the projections from data
Rinit           = zeros(NNeu,N_H);
Rinit_d         = zeros(NNeu,N_H);
R_t2            = zeros(NNeu,N_H);
xhat_init_true  = zeros(NVars,N_H); % initial x obtained using true decoder
x_t2            = zeros(NVars,N_H);
xhat_t2_true    = zeros(NVars,N_H);

for hh = 1:N_H
    Rinit(:,hh)             = rNeuMat(:,1,hh); % Just time-step 1 is recorded
    R_t2(:,hh)              = rNeuMat(:,2,hh); % Just time-step 2 is recorded
    Rinit_d(:,hh)           = rNeuMat_d(:,1,hh);
    x_t2(:,hh)              = xMat(:,2,hh);
    xhat_init_true(:,hh)    = V*rNeuMat(:,1,hh);
    xhat_t2_true(:,hh)      = V*rNeuMat(:,2,hh);
end

SigmoidH  = lam./(1 + exp(-hMat)); % sigmoid(hMat)
xinit     = SigmoidH;

HH      = [SigmoidH, 2*SigmoidH];
RR      = [Rinit, R_t2];


% Vhat = SigmoidH*pinv(Rinit);
Vhat = SigmoidH'\Rinit';

