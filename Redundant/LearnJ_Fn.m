function [JMat, CostVec, CostVec2, CostVec3]  = LearnJ_Fn(fMat, KVec, rMat, Jhat, nu, JMatTrue)

[NVars, N_T, N_H] = size(rMat);

% The provided value of Jhat is an initialization. Need to do gradient
% descent to improve it.

JhatNew     = zeros(NVars,NVars);
MaxIter     = 2000;
CostVec     = zeros(MaxIter+1,1);
CostVec2    = zeros(MaxIter,1);
CostVec3    = zeros(MaxIter,1);

CostVec(1)  = TAPCost(fMat,KVec,Jhat,rMat);

if ~isreal(CostVec(1)) || isinf(CostVec(1))
    keyboard;
end


for nIter = 1:MaxIter

% Pick one session for each mini-batch! One iteration is for one mini-batch

hh = mod(nIter-1,N_H)+1;

% First step is to compute gMat. This needs to be done for each iteration,
% because J keeps changing! 
% gMat is the counterpart of fMat: gMat = sum_J(W_ij)
% Size of gMat is (NVars,N_T-1)
    
GMat = zeros(NVars,N_T-1,27);

for kk = 0:26 %     
    abc = dec2base(kk,3,3);
    a = str2double(abc(1));
    b = str2double(abc(2));
    c = str2double(abc(3));

    for tt = 1:N_T-1
        r = rMat(:,tt,hh);
        for V = 1:NVars
            GMat(V,tt,kk+1) = KVec(kk+1)*(r(V).^b)*(Jhat(:,V).^a)'*(r.^c);
        end
    end
end

gMat = sum(GMat,3); clear GMat;

% Second step is to compute the eta matrix or eMat
% size of eMat is (NVars,NVars,N_T-1)

EMat = zeros(NVars,NVars,N_T-1,9);

for kk = 0:8 
    bc = dec2base(kk,3,2);
    b = str2double(bc(1));
    c = str2double(bc(2));
    idx1 = base2dec(['1',bc],3) + 1;
    idx2 = base2dec(['2',bc],3) + 1;
    
    for tt = 1:N_T-1
        r = rMat(:,tt,hh);
        EMat(:,:,tt,kk+1) = (KVec(idx1) + 2*KVec(idx2)*Jhat).*((r.^b)*(transpose(r).^c));
    end
end
    
eMat = sum(EMat,4); clear EMat;

for kk = 1:NVars
    dC_kk = 0;
    for tt = 1:N_T-1
        Wpart   = fMat(kk,tt,hh) - gMat(kk,tt);
        etapart = eMat(kk,kk,tt);

        dC_kk = dC_kk + Wpart*etapart;
    end
    dC_kk = -2*dC_kk;
    JhatNew(kk,kk) = Jhat(kk,kk) - nu*dC_kk; 
end

% Then do the off diagonal terms
for kk = 2:NVars
    for ll = 1:kk-1
        dC_kl = 0;
        for tt = 1:N_T-1
            Wpart1   = fMat(kk,tt,hh) - gMat(kk,tt);
            etapart1 = eMat(kk,ll,tt);
            Wpart2   = fMat(ll,tt,hh) - gMat(ll,tt);
            etapart2 = eMat(ll,kk,tt);
            dC_kl = dC_kl + Wpart1*etapart1 + Wpart2*etapart2;
        end
        dC_kl = -2*dC_kl;
        JhatNew(kk,ll) = Jhat(kk,ll) - nu*dC_kl; 
        JhatNew(ll,kk) = Jhat(ll,kk) - nu*dC_kl; 
    end
end    

% Check for convergence

% CostVec(nIter+1)  = TAPCost(fMat,KVec, JhatNew,rMat);
CostVec2(nIter) = norm(JhatNew(:) - JMatTrue(:));
CostVec3(nIter) = norm(JhatNew(:) - Jhat(:));

if CostVec(nIter+1) > CostVec(nIter)
   % keyboard;
end

% if norm(JhatNew(:) - Jhat(:)) < 1e-6
%     break;
% end

if CostVec2(nIter) < 0.1
   % break;
end

Jhat = JhatNew;


end % ----- iterations for loop

JMat = JhatNew;

CostVec     = CostVec(1:nIter+1);
CostVec2    = CostVec2(1:nIter);
CostVec3    = CostVec3(1:nIter);

