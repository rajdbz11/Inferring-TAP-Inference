function [dV,dW1,dW2,dB] = backprop(X,H,YD,V,W1,W2,B)

[NTrials, ~] = size(X);
Z = W1*X' + W2*H' + repmat(B,1,NTrials);
R = sigmoid(Z);
Y = V*R;
Y = Y';

dV  = (Y-YD)'*R';
dW1 = ((V'*(Y-YD)').*sigmoid(Z).*(1-sigmoid(Z)))*X;
dW2 = ((V'*(Y-YD)').*sigmoid(Z).*(1-sigmoid(Z)))*H;
dB  = sum((V'*(Y-YD)').*sigmoid(Z).*(1-sigmoid(Z)),2);