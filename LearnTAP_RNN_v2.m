% Script to embed TAP dynamics into neural response space as an
% RNN. After emedding the dynamics, the next step is to learn the embedding
% from data and subsequently learn the parameters of the inference
% algorithm.

clear;
load Data/KTrue;

NVars = 5; % No. of variables of x
JMat  = GenJMat(NVars); % Generate the coupling matrix J
JMat = JMat.*(ones(size(JMat))/3 + (1-1/3)*eye(NVars));
% JMat = JMat + diag((eps+abs(min(diag(JMat))))*ones(NVars,1));

JTrue       = JMat;
JTrueVec    = JMatToVec(JTrue);

% Parameters for TAP Dynamics
N_H = 1000; % No. of sessions. Each session has a different input
N_T = 30;  % No. of time steps per session
lam = 0.1; % Relaxation term

hMat = 1*randn(NVars, N_H); % Generate the inputs for each session

% Run the TAP dynamics
xMat = RunTAP(JMat, N_T, N_H, hMat, lam);

% Now construct the feedforward neural network

% gain constants for the hidden layer weights and bias terms

NNeu = 50*NVars; % No. of neurons in the hidden layer. We need a state expansion

g1 = 2/sqrt(NNeu);
g2 = 2/sqrt(NNeu);
g3 = 2/sqrt(NNeu);


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
    
    HidFull     = [HidFull, HiddenMat];
    OutFull     = [OutFull, xNextMat];

end

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
xhat_init_true  = zeros(NVars,N_H); % initial x obtained using true decoder

for hh = 1:N_H
    Rinit(:,hh)             = rNeuMat(:,1,hh); % Just time-step 1 is recorded
    Rinit_d(:,hh)           = rNeuMat_d(:,1,hh);
    xhat_init_true(:,hh)    = V*rNeuMat(:,1,hh);
end

SigmoidH  = lam./(1 + exp(-hMat)); % sigmoid(hMat)
Xinit     = SigmoidH;

% Xinit = Xinit + std(Xinit(:))/10*randn(size(Xinit));


save Data/Rinit Rinit
save Data/Xinit Xinit
save Data/rNeuMat rNeuMat
save Data/xMat xMat

Vhat = SigmoidH*pinv(Rinit);

% Check cost with true decoder
VVec = V(:);
[C, dVVec] = RNNInitCost(VVec);


options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-12, ...
    'MaxFunEvals',20000,'GradObj','on','TolFun',1e-12,'MaxIter',1000);


[Vhat2,fval,exitflag,output] = fminunc(@RNNFullCost,Vhat(:),options);
% [Vhat2,fval,exitflag,output] = fminunc(@RNNFullCost,Vhat2(:),options);


V2 = reshape(Vhat2,NVars,NNeu);

xhat = xMat*0;

for m = 1:N_H
    xhat(:,:,m) = V2*rNeuMat(:,:,m);  
end

CTRS{1} = linspace(0,1,80);
CTRS{2} = linspace(0,1,80);
Z =  hist3([xMat(:),xhat(:)],CTRS);
figure; imagesc(Z); colormap gray; colorbar


idx = randperm(N_H,20);

rMat = xMat;
fMat = GenfMat(rMat,hMat,lam);

rMat = rMat(:,:,idx);
fMat = fMat(:,:,idx);

save Data/rMat rMat; 
save Data/JMat JMat;
save Data/fMat fMat;

% GradDesCombined;
% ----------------------  using fmincon -----------------------------------

% % Setup constraints
% % M = N_H;
% M = 60;
% A = zeros(3*N_T*M,3*NNeu);
% 
% for m = 1:60
%     for t = 1:N_T
%         r = rNeuMat(:,t,m)';
%         tempMat = [r, r*0, r*0; r*0, r, r*0; r*0, r*0, r];
%         idx1 = (m-1)*N_T + t;
%         A((idx1 -1)*3 + 1:idx1*3,:) = tempMat;
%     end
% end
% 
% A = [A; -A];
% b = [ones(3*N_T*M,1); zeros(3*N_T*M,1)];
% 
% options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
% 
% [Vhat,fval,exitflag,output] = fmincon(@RNNInitCost,Vhat(:),A,b,[],[],-Inf,Inf,[],options);
