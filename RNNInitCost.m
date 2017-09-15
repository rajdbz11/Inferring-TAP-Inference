function [C, dVVec] = RNNInitCost(VVec)

addpath Data/
load Xinit; load Rinit; load rNeuMat; load xMat;

[NNeu, N_T, M] = size(rNeuMat);
[NVars,~] = size(Xinit);

V = reshape(VVec,NVars,NNeu);

Err = Xinit - V*Rinit;

C = sum(Err(:).^2)/M;



% Now for the regularization part
p   = 10;
aa  = 2;
bb  = 1;

alpha   = 1e-2; % regularization term weight 1e-2 works well

Reg     = 0;

for m = 1:M
    xdecoded    = V*rNeuMat(:,:,m);
    Reg_m       = (aa*xdecoded - bb).^p;
    Reg         = Reg + sum(Reg_m(:));

end
Reg = Reg/M/N_T;


C = C + alpha*Reg;


dVMat = V*0;

for m = 1:M
    
    ErrInit = Xinit(:,m) - V*Rinit(:,m); 
    dV_1    = -2*ErrInit*Rinit(:,m)';
    
    % In the regularization part
    dV_2 = 0;
    for t = 1:N_T % changed start time to 5 .. ignoring first few time points
        z = V*rNeuMat(:,t,m); 
        y = aa*z - bb; 
        dV_2 = dV_2 + p*(y.^(p-1))*aa*rNeuMat(:,t,m)'; 
    end
    dVMat = dVMat + dV_1/M + alpha*dV_2/M/N_T;
    
end

dVVec = dVMat(:);
