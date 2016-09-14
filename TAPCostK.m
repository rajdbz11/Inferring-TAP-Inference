function [C, dK] = TAPCostK(KVec)
addpath Data/
load rMat; load fMat; load JMat;

if nargin < 5
    nu = 1;
end

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
 
 
 JMat_p = ones(NVars,NVars,3);
 for pp = 1:2
     JMat_p(:,:,pp+1) = JMat.^pp;  
 end
 
 dK = zeros(27,1);
 C  = 0;
 
 
 for hh = 1:N_H % For each session
 
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
 
 % Computing the gradient
 tempMat = (fMat_h - gMat);
 tempMat = repmat(tempMat(:),27,1);
 X       = reshape(tempMat,NVars,N_T-1,27);
 Y       = X.*GamMat;
 dK_h    = sum(sum(Y,1),2);
 dK_h    = reshape(dK_h,27,1);
 dK_h    = -2*nu*dK_h;
 
 dK = dK + dK_h;
 C  = C + C_h;
 
 end
 
 