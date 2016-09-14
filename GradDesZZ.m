% Script to do gradient descent iteratively: fix J find K, fix K find J ..
% until we get good results! 
LearnTAP;

options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','tolX',1e-12,'MaxFunEvals',20000,'GradObj','on','TolFun',1e-12,'MaxIter',500);

Kinit = 0.1*randn(27,1);
temp  = 0.25*randn(NVars,NVars);
Jinit = temp'*temp;

KVec = Kinit;
JMat = Jinit;
save Data/KVec KVec;
save Data/JMat JMat;

for iter = 1:10
    KVec = fminunc(@TAPCostK,KVec,options);
    JMat = fminunc(@TAPCostJ,JMat,options);
end

figure; plot(KVec,'bx-'); hold on; plot(KTrue,'ro-');
figure; plot(JMat,'bx-'); hold on; plot(JTrueVec,'ro-');