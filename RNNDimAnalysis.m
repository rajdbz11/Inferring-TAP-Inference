function [dimX, dimR, rNeuMat, V, xhatMat,  ApproxErr, condV, normV] = RNNDimAnalysis(xMat,hMat,NNeu_factor)

[NVars,N_T,N_H] = size(xMat);

% Now construct the feedforward neural network

% gain constants for the hidden layer weights and bias terms
NNeu = NNeu_factor*NVars; % No. of neurons in the hidden layer. We need a state expansion
g1 = 2/sqrt(NNeu);
g2 = 2/sqrt(NNeu);
g3 = 2/sqrt(NNeu);


% Weight and biases
W1 = g1*randn(NNeu,NVars);
W2 = g2*randn(NNeu,NVars);
B  = g3*randn(NNeu,1);


% Concatenate the outputs and hidden layer activations for each session
% into one big matrix
OutFull = [];
HidFull = [];

 

for hh = 1:N_H
    InputMat    = repmat(hMat(:,hh), 1, N_T);
    xCurrMat    = [zeros(NVars,1), xMat(:,1:N_T-1,hh)]; % adding zeroinput also
    xNextMat    = xMat(:,:,hh);
    BiasMat     = repmat(B,1,N_T);
    SigInp      = W1*xCurrMat + W2*InputMat + BiasMat;
    HiddenMat   = 1./(1+exp(-SigInp));
    
    HidFull     = [HidFull, HiddenMat];
    OutFull     = [OutFull, xNextMat];

end

V = transpose(HidFull'\OutFull');

% Now generate the neural dynamics

rNeuMat     = zeros(NNeu,N_T,N_H);
xhatMat     = xMat*0;

for hh = 1:N_H
    rOld    = zeros(NNeu,1);
    for tt = 1:N_T
        SigInp  = W1*V*rOld + W2*hMat(:,hh) + B;
        rNew    = 1./(1+exp(-SigInp));
        
        rNeuMat(:,tt,hh) = rNew;
        rOld = rNew;
    end
    xhatMat(:,:,hh) = V*rNeuMat(:,:,hh);
end

covX = cov(transpose(reshape(xMat,NVars,N_T*N_H)));EigX = eig(covX); dimX = sum(EigX)^2/sum(EigX.^2);

covR = cov(transpose(reshape(rNeuMat,NNeu,N_T*N_H)));EigR = eig(covR); dimR = sum(EigR)^2/sum(EigR.^2);

ErrMat = (xMat - xhatMat).^2;

ApproxErr = sum(sum(ErrMat,1),2);
ApproxErr = ApproxErr(:);

condV = cond(V);

normV = norm(V,'fro');

