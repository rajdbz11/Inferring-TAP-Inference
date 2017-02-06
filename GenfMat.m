function fMat = GenfMat(rMat,hMat,lam)

[NVars,N_T,N_h] = size(rMat);
fMat            = zeros(NVars,N_T-1,N_h);
temp = [];
for hh = 1:N_h

    for tt = 1:N_T-1
        r   = rMat(:,tt  ,hh);
        rn  = rMat(:,tt+1,hh); % next time step
        y   = (rn - r*(1-lam))/lam;
        temp = [temp; y];
        % some kind of hack
        idx = find(y >= 1);
        y(idx) = 1 - 1e-6;
        idx = find(y <= 0);
        y(idx) = 1e-6;
        %%%
        fMat(:,tt,hh) = log(y./(1-y)) - hMat(:,hh);
    end

end
x = 0;

% fMat = zeros(NVars,T-1);
% for tt = 1:T-1
%     r   = rMat(:,tt);
%     rn  = rMat(:,tt+1); % next time step
%     y   = (rn - r*(1-lam))/lam;
%     fMat(:,tt) = log(y./(1-y)) - hMat(:,tt);
% end