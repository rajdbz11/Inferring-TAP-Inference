% Script to learn the parameters of the TAP Inference

clear;
load KTrue;

NVars = 5;
JMat  = GenJMat(NVars);
JTrue    = JMat;
JTrueVec = JMatToVec(JTrue);

% Parameters for TAP inf
N_T = 50;
N_H = 1;
lam = 0.1;

% options for learning
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-12,'MaxFunEvals',20000,'GradObj','on','TolFun',1e-12,'MaxIter',2000);

JhatMat = zeros(NVars*(NVars+1)/2,10);

for ni = 1:10

% Run the TAP Inference
hMat = randn(NVars, N_H);


rMat = RunTAP(JMat, N_T, N_H, hMat, lam);


% Generate the fMat
fMat = GenfMat(rMat,hMat,lam);

save rMat rMat; 
save JMat JMat;
save fMat fMat;


Kinit = 0.1*randn(27,1);
% temp  = 0.1*randn(NVars,NVars);
% Jinit = temp'*temp;
Jinit = JTrue;
Params = [Kinit; JMatToVec(Jinit)];

[Phat,fval,exitflag,output] = fminunc(@TAPCost3,Params,options);

Khat    = Phat(1:27);
JhatVec = Phat(28:end);


end