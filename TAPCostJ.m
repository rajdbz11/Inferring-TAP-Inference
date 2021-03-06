function [C, dJ] = TAPCostJ(Params)
addpath Data/
load rMat; load fMat; load KTrue;

if nargin < 5
    nu = 1;
end

KVec         = KTrue;
JMat         = JVecToMat(Params);


[NVars,N_T,N_H] = size(rMat);

LUT = ...
    [0     0     0
     0     0     1
     0     0     2
     0     1     0
     0     1     1
     0     1     2
     0     2     0
     0     2     1
     0     2     2
     1     0     0
     1     0     1
     1     0     2
     1     1     0
     1     1     1
     1     1     2
     1     2     0
     1     2     1
     1     2     2
     2     0     0
     2     0     1
     2     0     2
     2     1     0
     2     1     1
     2     1     2
     2     2     0
     2     2     1
     2     2     2];
 
 LUT2 = ...
   [0 0
    0 1
    0 2
    1 0
    1 1
    1 2
    2 0
    2 1
    2 2];
 
 
 JMat_p = ones(NVars,NVars,3);
 for pp = 1:2
     JMat_p(:,:,pp+1) = JMat.^pp;
 end
 
 C  = 0;
 dJ = 0;
 
 for hh = 1:N_H %  For each session
 
 rMat_p = ones(NVars,N_T,3);
 for pp = 1:2
     rMat_p(:,:,pp+1) = rMat(:,:,hh).^pp;
 end
     
 % Construct GMat
 GMat   = zeros(NVars,N_T-1,27);
 GamMat   = zeros(NVars,N_T-1,27);
 
 for kk = 1:27 
     a  = LUT(kk,1) + 1;
     b  = LUT(kk,2) + 1;
     c  = LUT(kk,3) + 1;
     Ja = JMat_p(:,:,a);
     Rb = rMat_p(:,:,b);
     Rc = rMat_p(:,:,c);
     
     for tt = 1:N_T-1
         
         rb = Rb(:,tt);
         rc = Rc(:,tt);
         
         for ii = 1:NVars
             GamMat(ii,tt,kk)   = rb(ii)*Ja(:,ii)'*rc;
             GMat(ii,tt,kk)     = KVec(kk)*rb(ii)*Ja(:,ii)'*rc;
         end
         
     end
 end
 
 gMat   = sum(GMat,3);
 fMat_h = fMat(:,:,hh);
 C_h    = sum((fMat_h(:) - gMat(:)).^2);
 
 % Construct etaMat
 etaMat    = zeros(NVars,NVars,N_T-1);
 
 for kk = 1:9
     
     b      = LUT2(kk,1) + 1;
     c      = LUT2(kk,2) + 1;
     Rb     = rMat_p(:,:,b);
     Rc     = rMat_p(:,:,c);
     K1bc   = KVec(kk+9);
     K2bc   = KVec(kk+18);
     
     for tt = 1:N_T-1
         rb = Rb(:,tt);
         rc = Rc(:,tt);
         etaMat(:,:,tt) = etaMat(:,:,tt) + (K1bc + 2*K2bc*JMat).*(rb*rc');
     end
 end

 
 % Now compute the gradient from each session
 dJ_h         = JMat*0;
 tempMat2   = fMat_h - gMat;
 
 for jj = 1:NVars
     for ii = jj:NVars
         for tt = 1:N_T-1
             if ii == jj
                 dJ_h(ii,jj) = dJ_h(ii,jj) + tempMat2(ii,tt)*etaMat(ii,ii,tt);
             else
                 dJ_h(ii,jj) = dJ_h(ii,jj) + tempMat2(ii,tt)*etaMat(ii,jj,tt) + tempMat2(jj,tt)*etaMat(jj,ii,tt);
             end
             
         end
     end
 end
 
 dJ_h = -2*nu*dJ_h;
 dJ_h = JMatToVec(dJ_h);
 
 % Update full cost and gradient for each session
 C  = C + C_h;
 dJ = dJ + dJ_h;
 
 
 end
 