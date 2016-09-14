% Script to do gradient descent type algo to find K, with J fixed

LearnTAP;

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-10,'MaxFunEvals',20000,'GradObj','on','TolFun',1e-10,'MaxIter',2000);

Kinit = 0.01*randn(27,1);

[Khat,fval,exitflag,output] = fminunc(@TAPCostK,Kinit,options);

figure; plot(Khat,'bx-'); hold on; plot(KTrue,'ro-');