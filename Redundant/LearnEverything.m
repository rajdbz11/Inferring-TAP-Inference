function [JMat, KVec, CostVec, CostVec2, CostVec3]  = LearnEverything(fMat, rMat, Kinit, Jinit, nu, JTrue, KTrue)

% Function to do gradient descent for both K and J terms

[NVars, N_T, N_H] = size(rMat);

JMatNew     = zeros(NVars,NVars);
JMat        = Jinit;

% KVecNew     = zeros(27,1);
KVec        = Kinit;

MaxIter     = 10; % No. of iterations of gradient descent
CostVec     = zeros(MaxIter+1,1);
CostVec2    = zeros(MaxIter,1);
CostVec3    = zeros(MaxIter,1);

CostVec(1)  = TAPCost(rMat,fMat,KVec,JMat);

if ~isreal(CostVec(1)) || isinf(CostVec(1))
    keyboard;
end



for nIter = 1:MaxIter % No of iterations
    
for hh = 1:N_H % No. of sessions

% First step is to compute gMat. 
% gMat is the counterpart of fMat: gMat = sum_J(W_ij)
% Size of gMat is (NVars,N_T-1)
    
GMat    = zeros(NVars,N_T-1,27);
GamMat  = zeros(NVars,N_T-1,27);

% KVec    = KTrue; % Just a debug hack to check the Learn JMat part
% KVecNew = KTrue;

JMat    = JTrue;
JMatNew = JTrue;

for kk = 0:26 %     
    
    abc = dec2base(kk,3,3);
    a   = str2double(abc(1));
    b   = str2double(abc(2));
    c   = str2double(abc(3));

    for tt = 1:N_T-1
        r = rMat(:,tt,hh);
        for V = 1:NVars
            GamMat(V,tt,kk+1)   = (r(V).^b)*(JMat(:,V).^a)'*(r.^c);
            GMat(V,tt,kk+1)     = KVec(kk+1)*GamMat(V,tt,kk+1);
        end
    end
end

gMat = sum(GMat,3); clear GMat;

if hh == 1
    dKVec = zeros(27,1);
end

for kk = 0:26 %     
    tempMat = (fMat(:,:,hh) - gMat).*GamMat(:,:,kk+1);
    dKVec(kk+1) = dKVec(kk+1) - 2*sum(sum(tempMat)); 
end

if hh == N_H
    KVecNew = KVec - nu*dKVec;
end

% Second step is to compute the eta matrix or eMat
% size of eMat is (NVars,NVars,N_T-1)

% EMat = zeros(NVars,NVars,N_T-1,9);
% 
% for kk = 0:8 
%     bc = dec2base(kk,3,2);
%     b = str2double(bc(1));
%     c = str2double(bc(2));
%     idx1 = base2dec(['1',bc],3) + 1;
%     idx2 = base2dec(['2',bc],3) + 1;
%     
%     for tt = 1:N_T-1
%         r = rMat(:,tt,hh);
%         EMat(:,:,tt,kk+1) = (KVec(idx1) + 2*KVec(idx2)*JMat).*((r.^b)*(transpose(r).^c));
%     end
% end
%     
% eMat = sum(EMat,4); clear EMat;
% 
% if hh == 1
%     dC_Mat = zeros(NVars,NVars);
% end
% 
% for kk = 1:NVars
%     
%     for tt = 1:N_T-1
%         Wpart   = fMat(kk,tt,hh) - gMat(kk,tt);
%         etapart = eMat(kk,kk,tt);
% 
%         dC_Mat(kk,kk) = dC_Mat(kk,kk) - 2*Wpart*etapart;
%     end
%     if hh == N_H
%         JMatNew(kk,kk) = JMat(kk,kk) - nu*dC_Mat(kk,kk); 
%     end
% end
% 
% % Then do the off diagonal terms
% for kk = 2:NVars
%     for ll = 1:kk-1
% 
%         for tt = 1:N_T-1
%             Wpart1   = fMat(kk,tt,hh) - gMat(kk,tt);
%             etapart1 = eMat(kk,ll,tt);
%             Wpart2   = fMat(ll,tt,hh) - gMat(ll,tt);
%             etapart2 = eMat(ll,kk,tt);
%             dC_Mat(kk,ll) = dC_Mat(kk,ll) - 2*(Wpart1*etapart1 + Wpart2*etapart2);
%         end
%         if hh == N_H
%             JMatNew(kk,ll) = JMat(kk,ll) - nu*dC_Mat(kk,ll); 
%             JMatNew(ll,kk) = JMat(ll,kk) - nu*dC_Mat(kk,ll); 
%         end
%     end
% end    



end % ----- sessions for loop

% JMat = JMatNew;
KVec = KVecNew;

% Check for convergence
% Params      = [JMatNew(:); KVecNew(:)];
% OldParams   = [JMat(:); KVec(:)];
% TrueParams  = [JTrue(:); KTrue(:)];
% CostVec(nIter+1) = TAPCost(rMat,fMat,KVecNew,JMatNew);
% CostVec2(nIter) = norm(Params - TrueParams);
% CostVec3(nIter) = norm(Params - OldParams);

% CostVec(nIter+1)= TAPCost(rMat,fMat,KTrue,JMat); % Using KTrue for now
% CostVec2(nIter) = norm(JMatNew(:) - JTrue(:));


CostVec(nIter+1)= TAPCost(rMat,fMat,KVec,JTrue); % Using JTrue for now
CostVec2(nIter) = norm(KVecNew(:) - KTrue(:));


end % ----- iterations for loop


% JMat = JMatNew;
KVec = KVecNew;
