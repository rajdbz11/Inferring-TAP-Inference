function rNeuMat = RunNeuTAP(JMat, N_T, N_H, hMat, U, lam, noisestd)

% Inputs: N_T = how long to run TAP inference for each stimulus h
%         N_h = No. of different sessions
%         lam = inference rate  

NVars = size(JMat,1);
NNeu  = size(U,1);
rNeuMat  = zeros(NNeu,N_T,N_H);

A = inv(U'*U)*U';
tempmat = [];

for hh = 1:N_H
    
    hVec    = hMat(:,hh);
    rOld    = 0*randn(NNeu,1); % Initialize 
    r       = zeros(NNeu,1);
    for nIter = 1:N_T
        SigVec = zeros(NVars,1);
        
        for V = 1:NVars
            xOld = A*rOld;
            inp = hVec(V) + 2*JMat(:,V)'*xOld + 4*(1-2*xOld(V))*(JMat(:,V).^2)'*(xOld.*(1-xOld)); %+ noisestd*randn(1,1);
            tempmat = [tempmat; inp];
            SigVec(V) = 1/(1+exp(-inp));
        end
        r = (1-lam)*rOld + lam*U*SigVec + noisestd*randn(NNeu,1);
        rNeuMat(:,nIter,hh) = r;
        rOld = r;
    end

end


