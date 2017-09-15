% Script to embed TAP dynamics into neural response space as an
% RNN. After emedding the dynamics, the next step is to learn the embedding
% from data and subsequently learn the parameters of the inference
% algorithm.

% Using backpropagation to construct the RNN that implicitly implements the
% TAP dynamics

%-------- First construct the dataset: i.e., the TAP dynamics -------------

clear;
rng(1);
load Data/KTrue;

NVars       = 3; % No. of variables of x
JMat        = GenJMat(NVars); % Generate the coupling matrix J
JTrue       = JMat;
JTrueVec    = JMatToVec(JTrue);

% Parameters for TAP Dynamics
N_H         = 1000; % No. of sessions. Each session has a different input
N_T         = 30;  % No. of time steps per session
lam         = 0.06; % Relaxation term

hMat        = 1*randn(NVars, N_H); % Generate the inputs for each session

% Run the TAP dynamics
xMat = RunTAP(JMat, N_T, N_H, hMat, lam);
% Include zeros as the first time step 
xMat_temp = zeros(NVars,N_T+1,N_H);
xMat_temp(:,2:end,:) = xMat;
xMat = xMat_temp; clear xMat_temp;

% Now to construct the dataset for the backprop
% X are the inputs
% Y are the desired outputs
X = xMat(:,1:end-1,:);
X = reshape(X,NVars,N_T*N_H);
X = X';

YD = xMat(:,2:end,:);
YD = reshape(YD,NVars,N_T*N_H);
YD = YD';

H = reshape(repmat(hMat,N_T,1),NVars,N_T*N_H);
H = H';




% -------------------------------------------------------------------------
% Setup the backpropagation code: this requires a function for the cost
% function, a function for doing the forward propagation and a function
% that computes the gradients to do the backpropagation

% Initialize the weights
NNeu    = 100*NVars; % No. of neurons in the hidden layer. We need a state expansion
W1      = 1/sqrt(NVars)*randn(NNeu,NVars);
W2      = 1/sqrt(NVars)*randn(NNeu,NVars);
B       = 1/sqrt(NVars)*randn(NNeu,1);
V       = 1/sqrt(NVars)*randn(NVars,NNeu);

C = CostBP(X,H,YD,V,W1,W2,B);


[NTrials, ~] = size(X);
Z = W1*X' + W2*H' + repmat(B,1,NTrials);
R = sigmoid(Z);


Vhat = transpose(R'\YD);

V = Vhat + 0.01*randn(size(V));

% compute the gradients now
Npasses = 20000;
CVec = zeros(Npasses,1);

for nn = 1:Npasses
    
% [NTrials, ~] = size(X);
% Z = W1*X' + W2*H' + repmat(B,1,NTrials);
% R = sigmoid(Z);
% Y = V*R;
% Y = Y';
% 
% dV = (Y-YD)'*R';
% dW1 = ((V'*(Y-YD)').*sigmoid(Z).*(1-sigmoid(Z)))*X;
% dW2 = ((V'*(Y-YD)').*sigmoid(Z).*(1-sigmoid(Z)))*H;
% dB = sum((V'*(Y-YD)').*sigmoid(Z).*(1-sigmoid(Z)),2);
idx = randperm(N_T*N_H,100);
[dV,dW1,dW2,dB] = backprop(X(idx,:),H(idx,:),YD(idx,:),V,W1,W2,B);

alp = 5e-5;
V   = V - alp*dV;
W1  = W1 - alp*dW1;
W2  = W1 - alp*dW2;
B   = B - alp*dB;



if mod(nn,500) == 0
    disp(nn);
    C2 = CostBP(X,H,YD,V,W1,W2,B);
    CVec(nn) = C2;
end

end

% Once the network is trained, run the RNN dynamics

% Now use the first time step to learn the projection matrix



