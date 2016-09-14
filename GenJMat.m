function JMat = GenJMat(NVars)
 
JSMat   = sprandsym(NVars,1);
JMat    = zeros(NVars,NVars);
for ii = 1:NVars
    for jj = 1:NVars
        JMat(ii,jj) = JSMat(ii,jj);
    end
end
clear JSMat;
eps = 0.1;
JMat = JMat + diag((eps+abs(min(diag(JMat))))*ones(NVars,1));
