
load KTrue;

NVars = 5;
% Generate or load the J matrix
JSMat   = sprandsym(NVars,1);
JMat    = zeros(NVars,NVars);
for ii = 1:NVars
    for jj = 1:NVars
        JMat(ii,jj) = JSMat(ii,jj);
    end
end
clear JSMat;
eps = 0.1;
JMat = JMat + diag((eps+abs(min(diag(JMat))))*ones(NVars,1));

load JMat2;

JTrue = JMat;
JTrueVec = [];
for kk = 1:NVars
    JTrueVec = [JTrueVec; JMat(kk:end,kk)]; 
end

NVars = size(JMat,1);


% Run the TAP Inference
N_T = 50;
N_H = 1;
lam = 0.1;

hMat = randn(NVars, N_H);

% load h1; load h2;
% hMat = [h1, h2];


rMat = RunTAP(JMat, N_T, N_H, hMat, lam);


% Generate the fMat
fMat = GenfMat(rMat,hMat,lam);

save rMat rMat; 
save JMat JMat;
save fMat fMat;