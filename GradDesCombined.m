% Script to do gradient descent type algo to find K and J together

%LearnTAP;
LearnTAP_WithEmbeddings;

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-12,'MaxFunEvals',20000,'GradObj','on','TolFun',1e-12,'MaxIter',2500);

Kinit = 0.1*randn(27,1);
temp  = 0.25*randn(NVars,NVars);
Jinit = temp'*temp;

Params = [Kinit; JMatToVec(Jinit)];

[Phat,fval,exitflag,output] = fminunc(@TAPCostCombined,Params,options);

Khat    = Phat(1:27);
JhatVec = Phat(28:end);

figure; plot(Khat,'bx-','LineWidth',2); hold on; plot(KTrue,'ro-','LineWidth',2);
figure; plot(JhatVec,'bx-','LineWidth',2); hold on; plot(JTrueVec,'ro-','LineWidth',2);