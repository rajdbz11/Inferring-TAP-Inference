function C = CostBP(X,H,YD,V,W1,W2,B)
% Function to compute cost for a 3 layer feedforward neural network
[NTrials, ~] = size(X);
Z = W1*X' + W2*H' + repmat(B,1,NTrials);
R = sigmoid(Z);
Y = V*R;
Y = Y';
C = (norm(Y(:) - YD(:))^2)/2;