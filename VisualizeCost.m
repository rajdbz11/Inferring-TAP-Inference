 LearnTAP;

% % Ki vs Kj
% DelV        = -0.1:0.01:0.1;
% [DM1,DM2]   = meshgrid(DelV,DelV);
% N           = length(DelV);
% 
% CostMat     = zeros(N,N);
% 
% for ii = 1:N
%     for jj   = 1:N
%         di   = DelV(ii);
%         dj   = DelV(jj);
%         K    = KTrue;
%         idxa = 1;
%         idxb = 2;
%         
%         K(idxa) = KTrue(idxa) + di;
%         K(idxb) = KTrue(idxb) + dj;
%         CostMat(ii,jj) =  TAPCost2(rMat,fMat,K,JMat);
%     end
% end
% 
% figure; surf(DM1,DM2,log10(CostMat+1e-3))
% 
% figure; contour(DM1,DM2,CostMat, 100); grid on


% % J vs K
% DelV        = -1:0.05:1;
% [DM1,DM2]   = meshgrid(DelV,DelV);
% N           = length(DelV);
%  
% CostMat     = zeros(N,N);
% 
% for ii = 1:N
%     for jj   = 1:N
%         di   = DelV(ii);
%         dj   = DelV(jj);
%         K    = KTrue;
%         J    = JTrue;
%         idxa = 25;
%         
%         %K(idxa) = KTrue(idxa) + di;
%         J(1,1)  = JTrue(1,1) + dj;
%         J(1,2)  = JTrue(1,2) + di;
%         CostMat(ii,jj) =  TAPCost2(rMat,fMat,K,J);
%     end
% end
% 
% figure; surf(DM1,DM2,log10(CostMat+1e-3))
% 
% figure; contour(DM1,DM2,CostMat, 100); grid on


% Single j direction
DelV        = -2:0.05:2;
Jinit       = 0.5*randn(NVars,NVars);
Jinit       = Jinit'*Jinit; 
N = length(DelV);
CostVec     = zeros(N,1);

for ii = 1:N
    di   = DelV(ii);
    K    = KTrue;
    J    = JTrue + DelV(ii)*Jinit;
    CostVec(ii) =  TAPCost2(rMat,fMat,K,J);
end
figure; plot(DelV,CostVec,'bx-')



% Kinit = 0.5*randn(27,1);
% Jinit = 0.5*randn(NVars,NVars);
% Jinit = Jinit'*Jinit;
% 
% aVec = -1:0.05:1;
% bVec = -1:0.05:1;
% [aMat,bMat] = meshgrid(aVec,bVec);
% 
% N =length(aVec);
% 
% CostMat = zeros(N,N);
% 
% for ii = 1:N
%     a = aVec(ii);
%     for jj = 1:N
%         b = bVec(jj);
% 
%         K = KTrue + a*Kinit;
%         J = JTrue + b*Jinit;
% 
%         CostMat(ii,jj) = TAPCost(rMat,fMat,K,J);
%     end
% end
% 
% figure; surf(aMat,bMat,CostMat)
% figure; surf(aMat,bMat,log10(CostMat+1e-4))
% figure; contour(aMat,bMat,CostMat,100)
