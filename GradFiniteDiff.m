LearnTAP;

Kinit = 0.1*randn(27,1);
temp  = 0.5*randn(NVars,NVars);
Jinit = temp'*temp;

JinitVec = JMatToVec(Jinit);
P = [Kinit; JinitVec];

dCVec = zeros(42,1);

eps = 1e-6;


C0 = TAPCost3(P);

for n = 1:42
    DelP = zeros(42,1); 
    DelP(n) = 1;
    dCVec(n) = TAPCost3(P+eps*DelP) - C0;
end

dCVecbyeps = dCVec/eps;
[C, dP] = TAPCost3(P);

% figure; plot(dP,'bx-')
% hold on
% plot(dCVecbyeps,'ro-')

figure; plot(dP(28:end),'bx-')
hold on
plot(dCVecbyeps(28:end),'ro-')