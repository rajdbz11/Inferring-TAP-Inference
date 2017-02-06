% Script to learn the parameters of the TAP Inference

clear;
load Data/KTrue;

NVars = 5;
JMat  = GenJMat(NVars);


JTrue = JMat;

JTrueVec = JMatToVec(JTrue);

% Run the TAP Inference
N_T = 30;
N_H = 5;
lam = 0.1;

hMat = 1*randn(NVars, N_H);


rMat = RunTAP(JMat, N_T, N_H, hMat, lam);



% Generate the fMat

fMat = GenfMat(rMat,hMat,lam);
if any(isinf(fMat(:)))
    disp('fMat has inf');
    keyboard;
end

save Data/rMat rMat; 
save Data/JMat JMat;
save Data/fMat fMat;