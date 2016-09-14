function rMat = RunTAP(JMat, N_T, N_h, hMat, lam)

% Inputs: N_T = how long to run TAP inference for each stimulus h
%         N_h = No. of different sessions
%         lam = inference rate  

NVars = size(JMat,1);

rMat  = zeros(NVars,N_T,N_h);

for hh = 1:N_h
    
    hVec    = hMat(:,hh);
    rOld    = 0.25*randn(NVars,1); % Initialize 
    r       = zeros(NVars,1);

    for nIter = 1:N_T
        for V = 1:NVars
            inp = hVec(V) + 2*JMat(:,V)'*rOld + 4*(1-2*rOld(V))*(JMat(:,V).^2)'*(rOld.*(1-rOld));
            r(V) = (1-lam)*rOld(V) + lam/(1+exp(-inp));
        end
        rMat(:,nIter,hh) = r;
        rOld = r;
    end

end

