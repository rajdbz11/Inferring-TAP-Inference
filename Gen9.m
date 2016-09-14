function C = Gen9(X)
N = length(X);
A = magic(N);
P = A'*A;
C = 0.5*X'*P*X;

