clear;
load Data/KTrue;

% Parameters for TAP Dynamics
NVars = 3; % No. of variables of x
N_H = 500; % No. of sessions. Each session has a different input
N_T = 30;  % No. of time steps per session
lam = 0.06; % Relaxation term


% load Data/JMat1
JMat  = GenJMat(NVars); % Generate the coupling matrix 
JMat = JMat.*(ones(size(JMat))/3 + (1-1/3)*eye(NVars));
hMat = 1*randn(NVars, N_H); % Generate the inputs for each session
JTrue       = JMat;
JTrueVec    = JMatToVec(JTrue);


% Run the TAP dynamics
xMat = RunTAP(JMat, N_T, N_H, hMat, lam);


% NNeu_factorVec = 50:25:200;
NNeu_factorVec = 20*ones(100,1);
L = length(NNeu_factorVec);
dimXVec = zeros(L,1);
dimRVec = zeros(L,1);
condVVec = zeros(L,1);
normVVec = zeros(L,1);
ErrMat  = zeros(N_H,L);


for kk = 1:L
    [dimX, dimR, rNeuMat, V, xhatMat, ApproxErr, condV, normV] = RNNDimAnalysis(xMat,hMat,NNeu_factorVec(kk));
    dimXVec(kk) = dimX;
    dimRVec(kk) = dimR;
    condVVec(kk) = condV;
    normVVec(kk) = normV;
    ErrMat(:,kk)  = ApproxErr;
end

disp(mean(ErrMat));