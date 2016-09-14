function fMat = GenfMat(rMat,hMat,lam)

[NVars,N_T,N_h] = size(rMat);
fMat            = zeros(NVars,N_T-1,N_h);

for hh = 1:N_h

    for tt = 1:N_T-1
        r   = rMat(:,tt  ,hh);
        rn  = rMat(:,tt+1,hh); % next time step
        y   = (rn - r*(1-lam))/lam;
        fMat(:,tt,hh) = log(y./(1-y)) - hMat(:,hh);
    end

end

% fMat = zeros(NVars,T-1);
% for tt = 1:T-1
%     r   = rMat(:,tt);
%     rn  = rMat(:,tt+1); % next time step
%     y   = (rn - r*(1-lam))/lam;
%     fMat(:,tt) = log(y./(1-y)) - hMat(:,tt);
% end