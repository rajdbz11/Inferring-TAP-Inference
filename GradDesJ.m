% Script to do gradient descent type algo to find J, given true K
LearnTAP;

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-12,'MaxFunEvals',20000,'GradObj','on','TolFun',1e-12,'MaxIter',2000);

temp  = 0.5*randn(NVars,NVars);
Jinit = temp'*temp;


Params = JMatToVec(Jinit);

[Phat,fval,exitflag,output] = fminunc(@TAPCostJ,Params,options);

JhatVec = Phat;

figure; plot(JhatVec,'bx-'); hold on; plot(JTrueVec,'ro-');
